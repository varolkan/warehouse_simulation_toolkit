"""2D geometry used everywhere else: poses, angles, polygons."""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple

Point = Tuple[float, float]


def normalize_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def angle_diff(target: float, source: float) -> float:
    """Shortest signed rotation that takes `source` to `target`."""
    return normalize_angle(target - source)


@dataclass(frozen=True)
class Pose2D:
    """A position and heading in a plane. Metres and radians."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.x, self.y, self.yaw))

    def distance_to(self, other: 'Pose2D') -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def advanced(self, forward: float, left: float = 0.0) -> 'Pose2D':
        """Return the pose `forward` metres ahead (and `left` metres to the left)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(self.x + c * forward - s * left, self.y + s * forward + c * left, self.yaw)

    def to_local(self, point: Point) -> Point:
        """Express a point given in this pose's parent frame in this pose's own frame."""
        dx, dy = point[0] - self.x, point[1] - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (c * dx + s * dy, -s * dx + c * dy)

    def to_parent(self, point: Point) -> Point:
        """Express a point given in this pose's own frame in its parent frame."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.x + c * point[0] - s * point[1], self.y + s * point[0] + c * point[1])

    def relative_to(self, origin: 'Pose2D') -> 'Pose2D':
        """Return this pose as seen from `origin`."""
        x, y = origin.to_local((self.x, self.y))
        return Pose2D(x, y, angle_diff(self.yaw, origin.yaw))


def polygon_area(polygon: Sequence[Point]) -> float:
    """Signed area (positive when counter-clockwise)."""
    area = 0.0
    for (x1, y1), (x2, y2) in zip(polygon, list(polygon[1:]) + [polygon[0]]):
        area += x1 * y2 - x2 * y1
    return area / 2.0


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Even-odd test. Points exactly on an edge count as inside."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if point_segment_distance(point, (x1, y1), (x2, y2)) < 1e-9:
            return True
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def point_segment_distance(point: Point, a: Point, b: Point) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_polygon_distance(point: Point, polygon: Sequence[Point]) -> float:
    """Distance to the polygon's outline; 0 inside it."""
    if point_in_polygon(point, polygon):
        return 0.0
    n = len(polygon)
    return min(point_segment_distance(point, polygon[i], polygon[(i + 1) % n]) for i in range(n))


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Whether segment ab properly crosses segment cd (shared endpoints do not count)."""
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = orient(c, d, a), orient(c, d, b)
    d3, d4 = orient(a, b, c), orient(a, b, d)
    return (d1 * d2 < 0.0) and (d3 * d4 < 0.0)


def polygon_is_simple(polygon: Sequence[Point]) -> bool:
    """No edge crosses another one."""
    n = len(polygon)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            if segments_intersect(
                    polygon[i], polygon[(i + 1) % n], polygon[j], polygon[(j + 1) % n]):
                return False
    return True


def rectangle(half_length: float, half_width: float) -> Tuple[Point, ...]:
    """Corners of a rectangle centred on the origin, counter-clockwise from front-left."""
    return ((half_length, half_width), (-half_length, half_width),
            (-half_length, -half_width), (half_length, -half_width))


def path_length(points: Iterable[Point]) -> float:
    total = 0.0
    previous = None
    for point in points:
        if previous is not None:
            total += math.hypot(point[0] - previous[0], point[1] - previous[1])
        previous = point
    return total
