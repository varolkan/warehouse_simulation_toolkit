"""The site file: stations and zones of one warehouse, validated when it is loaded.

A site file is YAML in the map frame, metres and degrees:

    name: warehouse
    frame: map
    stations:
      charger:
        kind: charger
        pose: [8.9, -8.0, -90]     # where the vehicle stands when docked, and its heading
      inbound:
        kind: transfer
        pose: [6.4, -1.45, 180]
        approach: 0.8              # optional: where the docking run starts, metres back
    zones:
      - name: frame_line
        kind: keepout
        polygon: [[-5.4, -2.0], [5.4, -2.0], [5.4, -1.0], [-5.4, -1.0]]
      - name: packing
        kind: speed
        limit: 0.4                 # m/s
        polygon: [[2.4, 4.0], [8.6, 4.0], [8.6, 5.3], [2.4, 5.3]]

A station's pose is the vehicle's pose when docked there; its heading points at whatever
the station is (a transfer stand, the charger contacts). The vehicle is brought by Nav2 to
the approach pose, `approach` metres straight back from there, and drives the rest itself.

Everything that can be checked without the map is checked on load, and every problem is
reported at once: a site file with a station inside a keep-out zone refuses to load rather
than fail on the first order that goes there.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agv_core.geometry import (
    Point, point_in_polygon, polygon_area, polygon_is_simple, Pose2D, rectangle,
    segments_intersect)

STATION_KINDS = ('transfer', 'charger')
ZONE_KINDS = ('keepout', 'speed')

DEFAULT_APPROACH = 0.8
MIN_APPROACH = 0.3
MAX_APPROACH = 3.0
MIN_STATION_SPACING = 0.5
MIN_ZONE_AREA = 0.01
MAX_SPEED_LIMIT = 2.0


class SiteError(ValueError):
    """The site file is wrong. The message lists every problem found."""

    def __init__(self, source: str, problems: Sequence[str]):
        self.source = source
        self.problems = list(problems)
        lines = '\n'.join(f'  - {p}' for p in self.problems)
        super().__init__(f'{source}: {len(self.problems)} problem(s) in the site file\n{lines}')


@dataclass(frozen=True)
class Station:
    """A place the vehicle docks at."""

    name: str
    kind: str
    pose: Pose2D
    approach: float = DEFAULT_APPROACH

    @property
    def approach_pose(self) -> Pose2D:
        """Where the docking run starts: `approach` metres straight back from the station."""
        return self.pose.advanced(-self.approach)

    @property
    def is_transfer(self) -> bool:
        return self.kind == 'transfer'

    @property
    def is_charger(self) -> bool:
        return self.kind == 'charger'


@dataclass(frozen=True)
class Zone:
    """An area with a rule: keep out, or drive no faster than `limit`."""

    name: str
    kind: str
    polygon: Tuple[Point, ...]
    limit: Optional[float] = None

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def crossed_by(self, a: Point, b: Point) -> bool:
        """Whether the straight line from a to b touches the zone."""
        if self.contains(a) or self.contains(b):
            return True
        n = len(self.polygon)
        return any(
            segments_intersect(a, b, self.polygon[i], self.polygon[(i + 1) % n]) for i in range(n))


class Site:
    """Stations and zones of one warehouse."""

    def __init__(self, name: str, frame: str, stations: Mapping[str, Station],
                 zones: Sequence[Zone], source: str = '<site>'):
        self.name = name
        self.frame = frame
        self.stations: Dict[str, Station] = dict(stations)
        self.zones: Tuple[Zone, ...] = tuple(zones)
        self.source = source

    # -- lookups ---------------------------------------------------------------------

    def station(self, name: str) -> Station:
        try:
            return self.stations[name]
        except KeyError:
            known = ', '.join(sorted(self.stations)) or 'none'
            raise KeyError(f"unknown station '{name}' (this site has: {known})") from None

    def has_station(self, name: str) -> bool:
        return name in self.stations

    @property
    def transfer_stations(self) -> List[Station]:
        return [s for s in self.stations.values() if s.is_transfer]

    @property
    def chargers(self) -> List[Station]:
        return [s for s in self.stations.values() if s.is_charger]

    def nearest_charger(self, pose: Pose2D) -> Optional[Station]:
        chargers = self.chargers
        if not chargers:
            return None
        return min(chargers, key=lambda s: pose.distance_to(s.pose))

    @property
    def keepout_zones(self) -> List[Zone]:
        return [z for z in self.zones if z.kind == 'keepout']

    @property
    def speed_zones(self) -> List[Zone]:
        return [z for z in self.zones if z.kind == 'speed']

    def keepout_at(self, point: Point) -> Optional[Zone]:
        """Return the keep-out zone a point lies in, if any."""
        for zone in self.keepout_zones:
            if zone.contains(point):
                return zone
        return None

    def speed_limit_at(self, point: Point) -> Optional[float]:
        """Return the lowest speed limit that applies at a point, or None if there is none."""
        limits = [z.limit for z in self.speed_zones if z.contains(point)]
        return min(limits) if limits else None

    def station_at(self, pose: Pose2D, tolerance: float = 0.3) -> Optional[Station]:
        """Return the station whose docked pose is within `tolerance` of a pose, if any."""
        best = None
        for station in self.stations.values():
            distance = pose.distance_to(station.pose)
            if distance <= tolerance and (best is None or distance < pose.distance_to(best.pose)):
                best = station
        return best

    # -- checks that need the vehicle ---------------------------------------------------

    def vehicle_problems(self, half_length: float, half_width: float) -> List[str]:
        """Stations where the docked vehicle's outline would reach into a keep-out zone."""
        problems = []
        for station in self.stations.values():
            for corner in rectangle(half_length, half_width):
                zone = self.keepout_at(station.pose.to_parent(corner))
                if zone is not None:
                    problems.append(
                        f"docked at '{station.name}' the vehicle reaches into keep-out zone "
                        f"'{zone.name}'")
                    break
        return problems

    # -- loading ---------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Any, source: str = '<site>') -> 'Site':
        problems: List[str] = []
        if not isinstance(data, Mapping):
            raise SiteError(source, ['the file must be a mapping with name, stations and zones'])

        name = data.get('name')
        if not isinstance(name, str) or not name.strip():
            problems.append("'name' is missing or empty")
            name = ''
        frame = data.get('frame', 'map')
        if not isinstance(frame, str) or not frame.strip():
            problems.append("'frame' must be a frame name")
            frame = 'map'

        unknown = sorted(set(data) - {'name', 'frame', 'stations', 'zones'})
        if unknown:
            problems.append(f"unknown top-level key(s): {', '.join(map(str, unknown))}")

        stations = _parse_stations(data.get('stations'), problems)
        zones = _parse_zones(data.get('zones', []), problems)
        _cross_check(stations, zones, problems)

        if problems:
            raise SiteError(source, problems)
        return cls(name, frame, stations, zones, source)

    @classmethod
    def load(cls, path: str) -> 'Site':
        import yaml  # deferred: only loading from a file needs it

        try:
            with open(path) as fh:
                data = yaml.safe_load(fh)
        except OSError as error:
            raise SiteError(path, [f'cannot read the file: {error.strerror}']) from None
        except yaml.YAMLError as error:
            raise SiteError(path, [f'not valid YAML: {error}']) from None
        return cls.from_dict(data, source=path)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _parse_stations(raw: Any, problems: List[str]) -> Dict[str, Station]:
    stations: Dict[str, Station] = {}
    if not isinstance(raw, Mapping) or not raw:
        problems.append("'stations' must be a mapping of at least one station")
        return stations
    for name, entry in raw.items():
        label = f"station '{name}'"
        if not isinstance(name, str) or not name or any(c.isspace() for c in name):
            problems.append(f'{label}: a station name is one word without spaces')
            continue
        if not isinstance(entry, Mapping):
            problems.append(f'{label}: must be a mapping with kind and pose')
            continue
        unknown = sorted(set(entry) - {'kind', 'pose', 'approach'})
        if unknown:
            problems.append(f"{label}: unknown key(s): {', '.join(map(str, unknown))}")
        kind = entry.get('kind')
        if kind not in STATION_KINDS:
            problems.append(
                f"{label}: kind is '{kind}', must be one of {', '.join(STATION_KINDS)}")
            continue
        pose_raw = entry.get('pose')
        values = [_number(v) for v in pose_raw] if isinstance(pose_raw, (list, tuple)) else []
        if len(values) != 3 or any(v is None for v in values):
            problems.append(f'{label}: pose must be [x, y, heading in degrees]')
            continue
        approach = _number(entry.get('approach', DEFAULT_APPROACH))
        if approach is None or not MIN_APPROACH <= approach <= MAX_APPROACH:
            problems.append(
                f'{label}: approach must be between {MIN_APPROACH} and {MAX_APPROACH} m')
            continue
        stations[name] = Station(
            name, kind, Pose2D(values[0], values[1], math.radians(values[2])), approach)
    return stations


def _parse_zones(raw: Any, problems: List[str]) -> List[Zone]:
    zones: List[Zone] = []
    if raw is None:
        return zones
    if not isinstance(raw, (list, tuple)):
        problems.append("'zones' must be a list")
        return zones
    seen = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            problems.append(f'zone {index + 1}: must be a mapping with name, kind and polygon')
            continue
        name = entry.get('name')
        label = f"zone '{name}'" if isinstance(name, str) and name else f'zone {index + 1}'
        if not isinstance(name, str) or not name:
            problems.append(f'{label}: needs a name')
            continue
        if name in seen:
            problems.append(f'{label}: the name is used twice')
            continue
        seen.add(name)
        unknown = sorted(set(entry) - {'name', 'kind', 'polygon', 'limit'})
        if unknown:
            problems.append(f"{label}: unknown key(s): {', '.join(map(str, unknown))}")
        kind = entry.get('kind')
        if kind not in ZONE_KINDS:
            problems.append(f"{label}: kind is '{kind}', must be one of {', '.join(ZONE_KINDS)}")
            continue
        polygon = _parse_polygon(entry.get('polygon'))
        if polygon is None:
            problems.append(f'{label}: polygon must be a list of at least three [x, y] points')
            continue
        if not polygon_is_simple(polygon):
            problems.append(f'{label}: the polygon crosses itself')
            continue
        if abs(polygon_area(polygon)) < MIN_ZONE_AREA:
            problems.append(f'{label}: the polygon encloses no area')
            continue
        limit = None
        if kind == 'speed':
            limit = _number(entry.get('limit'))
            if limit is None or not 0.0 < limit <= MAX_SPEED_LIMIT:
                problems.append(
                    f'{label}: a speed zone needs a limit between 0 and {MAX_SPEED_LIMIT} m/s')
                continue
        elif 'limit' in entry:
            problems.append(f'{label}: a keep-out zone has no limit')
            continue
        zones.append(Zone(name, kind, polygon, limit))
    return zones


def _parse_polygon(raw: Any) -> Optional[Tuple[Point, ...]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    points = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        x, y = _number(item[0]), _number(item[1])
        if x is None or y is None:
            return None
        points.append((x, y))
    return tuple(points)


def _cross_check(stations: Mapping[str, Station], zones: Sequence[Zone],
                 problems: List[str]) -> None:
    keepouts = [z for z in zones if z.kind == 'keepout']
    names = list(stations)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            spacing = stations[first].pose.distance_to(stations[second].pose)
            if spacing < MIN_STATION_SPACING:
                problems.append(
                    f"stations '{first}' and '{second}' are {spacing:.2f} m apart; stations "
                    f'need {MIN_STATION_SPACING} m between them')
    for station in stations.values():
        here = (station.pose.x, station.pose.y)
        start = (station.approach_pose.x, station.approach_pose.y)
        for zone in keepouts:
            if zone.contains(here):
                problems.append(f"station '{station.name}' is inside keep-out zone '{zone.name}'")
            elif zone.crossed_by(start, here):
                problems.append(
                    f"the docking run into station '{station.name}' ({station.approach} m) "
                    f"crosses keep-out zone '{zone.name}'")
