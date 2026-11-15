"""Occupancy grids: reading and writing map files, clearance, ray casting and A*.

A GridMap holds what a Nav2 map file holds (map.yaml + map.pgm) as an array with row 0 at
the bottom, so `data[row, col]` is the cell at `origin + (col, row) * resolution`.

Nothing here is used by the vehicle to navigate: Nav2 does that. The grid serves the
tools and tests around it: the keep-out and speed masks are drawn on it, the site checks
measure station clearance on it, the simulated scanner casts its rays through it, and the
navigator that stands in for Nav2 in unit tests plans on it.
"""

import heapq
import math
import os
from typing import List, Optional, Sequence, Tuple

from agv_core.geometry import Point
import numpy as np

FREE = 0
OCCUPIED = 100
UNKNOWN = -1

# Grey levels map_server writes for free, occupied and unknown cells.
PGM_FREE = 254
PGM_OCCUPIED = 0
PGM_UNKNOWN = 205

# Thresholds on "how occupied" a grey level is, (255 - grey) / 255. Unknown's 205 comes out
# at 0.196, a hair above FREE_THRESH: with the 0.25 that is often written instead, every
# unmapped cell of a saved map loads as free floor.
OCCUPIED_THRESH = 0.65
FREE_THRESH = 0.196


class MapError(ValueError):
    """A map file could not be read."""


class GridMap:
    """An occupancy grid in a metric frame."""

    def __init__(self, data: np.ndarray, resolution: float, origin: Point = (0.0, 0.0)):
        if data.ndim != 2:
            raise MapError('grid data must be two-dimensional')
        if not resolution > 0.0:
            raise MapError('resolution must be positive')
        self.data = np.asarray(data, dtype=np.int16)
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))

    # -- geometry --------------------------------------------------------------------

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """x_min, y_min, x_max, y_max of the area the grid covers."""
        return (self.origin[0], self.origin[1],
                self.origin[0] + self.width * self.resolution,
                self.origin[1] + self.height * self.resolution)

    def cell_of(self, x: float, y: float) -> Tuple[int, int]:
        """(col, row) of the cell containing a point. May lie outside the grid."""
        return (int(math.floor((x - self.origin[0]) / self.resolution)),
                int(math.floor((y - self.origin[1]) / self.resolution)))

    def centre_of(self, col: int, row: int) -> Point:
        return (self.origin[0] + (col + 0.5) * self.resolution,
                self.origin[1] + (row + 0.5) * self.resolution)

    def contains_cell(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def value_at(self, x: float, y: float) -> int:
        """Cell value at a point; UNKNOWN outside the grid."""
        col, row = self.cell_of(x, y)
        if not self.contains_cell(col, row):
            return UNKNOWN
        return int(self.data[row, col])

    def is_free(self, x: float, y: float) -> bool:
        return self.value_at(x, y) == FREE

    def like(self, fill: int = FREE) -> 'GridMap':
        """Return an empty grid with the same geometry."""
        return GridMap(
            np.full(self.data.shape, fill, dtype=np.int16), self.resolution, self.origin)

    # -- obstacles and clearance --------------------------------------------------------

    @property
    def occupied(self) -> np.ndarray:
        return self.data >= OCCUPIED

    def blocked(self, unknown_is_blocked: bool = True) -> np.ndarray:
        """Cells a vehicle's centre may never be in, before any clearance is applied."""
        mask = self.occupied
        if unknown_is_blocked:
            mask = mask | (self.data == UNKNOWN)
        return mask

    def inflated(self, radius: float, unknown_is_blocked: bool = True) -> np.ndarray:
        """Cells whose centre is within `radius` of a blocked cell."""
        return _dilate(self.blocked(unknown_is_blocked), radius / self.resolution)

    def clearance_at(self, x: float, y: float, search: float = 3.0) -> float:
        """Distance from a point to the nearest occupied cell, at most `search`."""
        col, row = self.cell_of(x, y)
        reach = int(math.ceil(search / self.resolution)) + 1
        c0, c1 = max(0, col - reach), min(self.width, col + reach + 1)
        r0, r1 = max(0, row - reach), min(self.height, row + reach + 1)
        if c0 >= c1 or r0 >= r1:
            return search
        rows, cols = np.nonzero(self.occupied[r0:r1, c0:c1])
        if rows.size == 0:
            return search
        # Distance to the nearest edge of each occupied cell, not to its centre.
        cx = self.origin[0] + (cols + c0 + 0.5) * self.resolution
        cy = self.origin[1] + (rows + r0 + 0.5) * self.resolution
        half = self.resolution / 2.0
        dx = np.maximum(np.abs(cx - x) - half, 0.0)
        dy = np.maximum(np.abs(cy - y) - half, 0.0)
        return float(min(search, np.hypot(dx, dy).min()))

    def fill_polygon(self, polygon: Sequence[Point], value: int) -> int:
        """Set every cell whose centre is inside a polygon. Returns the number of cells set."""
        mask = self.polygon_mask(polygon)
        self.data[mask] = value
        return int(mask.sum())

    def polygon_mask(self, polygon: Sequence[Point]) -> np.ndarray:
        """Boolean array: cells whose centre is inside a polygon (even-odd rule)."""
        xs = self.origin[0] + (np.arange(self.width) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(self.height) + 0.5) * self.resolution
        px, py = np.meshgrid(xs, ys)
        inside = np.zeros(self.data.shape, dtype=bool)
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if y1 == y2:
                continue
            crosses = (y1 > py) != (y2 > py)
            x_at = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (px < x_at)
        return inside

    # -- ray casting -------------------------------------------------------------------

    def ray_cast(self, x: float, y: float, angles: np.ndarray, max_range: float,
                 step: Optional[float] = None) -> np.ndarray:
        """Range to the first occupied cell along each ray; +inf where none within reach.

        Rays are sampled every `step` (half a cell by default), so a hit is reported up
        to one step short of the surface. Unknown cells and space outside the grid are
        see-through.
        """
        step = step or self.resolution / 2.0
        distances = np.arange(step, max_range + step / 2.0, step)
        angles = np.asarray(angles, dtype=float)
        px = x + np.outer(np.cos(angles), distances)
        py = y + np.outer(np.sin(angles), distances)
        cols = np.floor((px - self.origin[0]) / self.resolution).astype(np.int64)
        rows = np.floor((py - self.origin[1]) / self.resolution).astype(np.int64)
        inside = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
        hit = np.zeros(px.shape, dtype=bool)
        hit[inside] = self.occupied[rows[inside], cols[inside]]
        first = hit.argmax(axis=1)
        ranges = np.where(hit.any(axis=1), distances[first], np.inf)
        return ranges

    def line_is_clear(self, a: Point, b: Point, blocked: np.ndarray) -> bool:
        """Whether the straight line from a to b stays off the cells set in `blocked`."""
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        samples = max(2, int(math.ceil(length / (self.resolution / 2.0))) + 1)
        for t in np.linspace(0.0, 1.0, samples):
            col, row = self.cell_of(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if not self.contains_cell(col, row) or blocked[row, col]:
                return False
        return True

    # -- planning ----------------------------------------------------------------------

    def plan(self, start: Point, goal: Point, clearance: float,
             unknown_is_blocked: bool = True,
             extra_blocked: Optional[np.ndarray] = None) -> Optional[List[Point]]:
        """Plan from start to goal with A*, the centre `clearance` from every blocked cell.

        Returns the route as a short list of points (start, corners, goal), or None when
        there is no route. `extra_blocked` adds cells to avoid, such as keep-out zones.
        """
        blocked = self.inflated(clearance, unknown_is_blocked)
        if extra_blocked is not None:
            blocked = blocked | extra_blocked
        start_cell, goal_cell = self.cell_of(*start), self.cell_of(*goal)
        for cell in (start_cell, goal_cell):
            if not self.contains_cell(*cell):
                return None
        # The vehicle may be standing closer to something than `clearance` (docked at a
        # station): let it leave from, and arrive at, the nearest cell that is clear.
        start_cell = _nearest_clear(blocked, start_cell, limit=int(0.8 / self.resolution))
        goal_cell = _nearest_clear(blocked, goal_cell, limit=int(0.8 / self.resolution))
        if start_cell is None or goal_cell is None:
            return None
        cells = _astar(blocked, start_cell, goal_cell)
        if cells is None:
            return None
        points = [start] + [self.centre_of(c, r) for c, r in cells[1:-1]] + [goal]
        return self._shortcut(points, blocked)

    def _shortcut(self, points: List[Point], blocked: np.ndarray) -> List[Point]:
        """Drop every point that the straight line between its neighbours makes needless."""
        if len(points) <= 2:
            return points
        result = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            reach = len(points) - 1
            while (reach > anchor + 1
                   and not self.line_is_clear(points[anchor], points[reach], blocked)):
                reach -= 1
            result.append(points[reach])
            anchor = reach
        return result

    # -- files -------------------------------------------------------------------------

    @classmethod
    def load(cls, yaml_path: str) -> 'GridMap':
        """Read a Nav2 map (yaml + pgm), thresholded the way map_server does."""
        import yaml  # deferred: only file access needs it

        try:
            with open(yaml_path) as fh:
                meta = yaml.safe_load(fh)
        except OSError as error:
            raise MapError(f'{yaml_path}: {error.strerror}') from None
        if not isinstance(meta, dict):
            raise MapError(f'{yaml_path}: not a map description')
        for key in ('image', 'resolution', 'origin'):
            if key not in meta:
                raise MapError(f"{yaml_path}: '{key}' is missing")
        image_path = meta['image']
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image_path)
        pixels = read_pgm(image_path)[::-1].astype(np.float64)   # file row 0 is the top
        mode = str(meta.get('mode', 'trinary')).lower()
        if mode == 'raw':
            data = np.where(pixels <= OCCUPIED, pixels, UNKNOWN).astype(np.int16)
        else:
            negate = int(meta.get('negate', 0))
            occupancy = pixels / 255.0 if negate else (255.0 - pixels) / 255.0
            occupied_thresh = float(meta.get('occupied_thresh', OCCUPIED_THRESH))
            free_thresh = float(meta.get('free_thresh', FREE_THRESH))
            data = np.full(pixels.shape, UNKNOWN, dtype=np.int16)
            data[occupancy > occupied_thresh] = OCCUPIED
            data[occupancy < free_thresh] = FREE
            if mode == 'scale':
                between = data == UNKNOWN
                data[between] = np.rint(
                    99.0 * (occupancy[between] - free_thresh) / (occupied_thresh - free_thresh)
                ).astype(np.int16)
        origin = meta['origin']
        if abs(float(origin[2]) if len(origin) > 2 else 0.0) > 1e-9:
            raise MapError(f'{yaml_path}: rotated map origins are not supported')
        return cls(data, float(meta['resolution']), (float(origin[0]), float(origin[1])))

    def save(self, yaml_path: str, mode: str = 'trinary') -> None:
        """Write the grid as a Nav2 map: `<name>.yaml` next to `<name>.pgm`.

        `trinary` writes free / occupied / unknown; `raw` writes the cell values as grey
        levels unchanged (0..100), which is what a speed-limit mask needs.
        """
        if mode == 'raw':
            pixels = np.where(self.data < 0, 255, np.clip(self.data, 0, OCCUPIED))
        elif mode == 'trinary':
            pixels = np.full(self.data.shape, PGM_UNKNOWN)
            pixels[self.data == FREE] = PGM_FREE
            pixels[self.data >= OCCUPIED] = PGM_OCCUPIED
        else:
            raise MapError(f"unknown map mode '{mode}'")
        base = os.path.splitext(yaml_path)[0]
        write_pgm(base + '.pgm', pixels[::-1].astype(np.uint8))
        lines = [
            f'image: {os.path.basename(base)}.pgm',
            f'mode: {mode}',
            f'resolution: {self.resolution:.3f}',
            f'origin: [{self.origin[0]:.3f}, {self.origin[1]:.3f}, 0]',
            'negate: 0',
            f'occupied_thresh: {OCCUPIED_THRESH}',
            f'free_thresh: {FREE_THRESH}',
        ]
        with open(yaml_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')


def read_pgm(path: str) -> np.ndarray:
    """Read a binary (P5) or plain (P2) PGM into an array, top row first."""
    try:
        with open(path, 'rb') as fh:
            raw = fh.read()
    except OSError as error:
        raise MapError(f'{path}: {error.strerror}') from None
    tokens: List[bytes] = []
    position = 0
    while len(tokens) < 4:
        while position < len(raw) and raw[position:position + 1].isspace():
            position += 1
        if position >= len(raw):
            raise MapError(f'{path}: truncated PGM header')
        if raw[position:position + 1] == b'#':
            while position < len(raw) and raw[position:position + 1] != b'\n':
                position += 1
            continue
        start = position
        while position < len(raw) and not raw[position:position + 1].isspace():
            position += 1
        tokens.append(raw[start:position])
    magic = tokens[0]
    try:
        width, height, maximum = (int(t) for t in tokens[1:4])
    except ValueError:
        raise MapError(f'{path}: bad PGM header') from None
    if magic == b'P5':
        position += 1   # the single whitespace byte that ends the header
        dtype = np.dtype('>u2') if maximum > 255 else np.dtype('u1')
        count = width * height
        body = np.frombuffer(raw, dtype=dtype, count=count, offset=position) \
            if len(raw) - position >= count * dtype.itemsize else None
    elif magic == b'P2':
        values = raw[position:].split()
        body = np.array([int(v) for v in values[:width * height]]) \
            if len(values) >= width * height else None
    else:
        raise MapError(f'{path}: not a PGM image')
    if body is None:
        raise MapError(f'{path}: image data is shorter than {width}x{height}')
    pixels = body.reshape(height, width).astype(np.float64)
    if maximum != 255:
        pixels = pixels * 255.0 / maximum
    return np.rint(pixels).astype(np.int16)


def write_pgm(path: str, pixels: np.ndarray) -> None:
    height, width = pixels.shape
    with open(path, 'wb') as fh:
        fh.write(f'P5\n{width} {height}\n255\n'.encode('ascii'))
        fh.write(np.ascontiguousarray(pixels, dtype=np.uint8).tobytes())


def _dilate(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    """Grow a boolean mask by a disc. Cells outside the array count as not set."""
    # 0.30 m of 0.05 m cells is 5.999999999999999 of them: do not lose a ring to rounding.
    radius_cells += 1e-9
    reach = int(math.floor(radius_cells))
    if reach <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.zeros((height + 2 * reach, width + 2 * reach), dtype=bool)
    padded[reach:reach + height, reach:reach + width] = mask
    out = np.zeros_like(mask)
    limit = radius_cells * radius_cells
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            if dr * dr + dc * dc <= limit:
                out |= padded[reach + dr:reach + dr + height, reach + dc:reach + dc + width]
    return out


def _nearest_clear(blocked: np.ndarray, cell: Tuple[int, int],
                   limit: int) -> Optional[Tuple[int, int]]:
    col, row = cell
    if not blocked[row, col]:
        return cell
    height, width = blocked.shape
    best, best_distance = None, None
    for dr in range(-limit, limit + 1):
        for dc in range(-limit, limit + 1):
            r, c = row + dr, col + dc
            if 0 <= r < height and 0 <= c < width and not blocked[r, c]:
                distance = dr * dr + dc * dc
                if best_distance is None or distance < best_distance:
                    best, best_distance = (c, r), distance
    return best


_NEIGHBOURS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
               (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)))


def _astar(blocked: np.ndarray, start: Tuple[int, int],
           goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """8-connected A* over cells that are not blocked. Cells are (col, row)."""
    height, width = blocked.shape

    def heuristic(col: int, row: int) -> float:
        dc, dr = abs(col - goal[0]), abs(row - goal[1])
        return (dc + dr) + (math.sqrt(2.0) - 2.0) * min(dc, dr)

    best = {start: 0.0}
    parent = {}
    queue = [(heuristic(*start), 0.0, start)]
    closed = set()
    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cell in closed:
            continue
        if cell == goal:
            route = [cell]
            while cell in parent:
                cell = parent[cell]
                route.append(cell)
            return route[::-1]
        closed.add(cell)
        col, row = cell
        for dc, dr, step in _NEIGHBOURS:
            c, r = col + dc, row + dr
            if not (0 <= c < width and 0 <= r < height) or blocked[r, c]:
                continue
            # No squeezing diagonally between two blocked cells.
            if dc and dr and (blocked[row, c] or blocked[r, col]):
                continue
            candidate = cost + step
            if candidate < best.get((c, r), math.inf):
                best[(c, r)] = candidate
                parent[(c, r)] = cell
                heapq.heappush(queue, (candidate + heuristic(c, r), candidate, (c, r)))
    return None
