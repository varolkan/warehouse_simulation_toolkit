"""Keep-out and speed-limit masks for Nav2, drawn from the site file onto the map's grid.

Nav2's costmap filters read a zone as an image the size of the map. The site file is the
one place zones are written down; these functions turn it into the two images and check
the site against the map while they are at it. The masks are files in the repository, so
the test suite regenerates them and fails if a zone was edited and the masks were not.

Keep-out mask: occupied where the vehicle's centre may not be.
Speed mask:    cell value v means "no faster than v x SPEED_STEP m/s here"; 0 is no limit.
"""

from typing import List

from agv_core.grid import FREE, GridMap, OCCUPIED
from agv_core.sites import Site

# One grey level of the speed mask, in m/s. nav2.yaml's speed_filter_info_server has the
# same number as its `multiplier`.
SPEED_STEP = 0.01


def keepout_mask(site: Site, grid: GridMap) -> GridMap:
    """Return a grid like `grid`, occupied inside every keep-out zone and free elsewhere."""
    mask = grid.like(FREE)
    for zone in site.keepout_zones:
        mask.fill_polygon(zone.polygon, OCCUPIED)
    return mask


def speed_mask(site: Site, grid: GridMap) -> GridMap:
    """Return a grid like `grid` holding each cell's speed limit in steps of SPEED_STEP."""
    mask = grid.like(0)
    for zone in sorted(site.speed_zones, key=lambda z: -z.limit):
        # Slowest last: where zones overlap, the lower limit wins.
        mask.fill_polygon(zone.polygon, speed_value(zone.limit))
    return mask


def speed_value(limit: float) -> int:
    """Return the mask value for a limit in m/s (1..100)."""
    return max(1, min(OCCUPIED, int(round(limit / SPEED_STEP))))


def site_map_problems(site: Site, grid: GridMap, half_length: float, half_width: float,
                      docking_gap: float = 0.10) -> List[str]:
    """List what is wrong with a site on a given map. An empty list means nothing is.

    Checked for every station:
      - the docked pose and the approach pose are on mapped, free floor;
      - the approach pose has room to turn on the spot (the vehicle arrives there from any
        direction and turns to face the station);
      - the docking run between the two is free on both sides of the vehicle;
      - every zone lies at least partly on the map.
    """
    problems: List[str] = []
    turning_radius = (half_length ** 2 + half_width ** 2) ** 0.5
    for station in site.stations.values():
        pose, start = station.pose, station.approach_pose
        if not grid.is_free(pose.x, pose.y):
            problems.append(f"station '{station.name}' is not on free, mapped floor")
            continue
        if not grid.is_free(start.x, start.y):
            problems.append(f"the approach pose of station '{station.name}' is not on free, "
                            'mapped floor')
            continue
        room = grid.clearance_at(start.x, start.y)
        if room < turning_radius + 0.05:
            problems.append(
                f"the approach pose of station '{station.name}' is {room:.2f} m from the "
                f'nearest obstacle; turning on the spot needs {turning_radius + 0.05:.2f} m')
        # Sides of the docking run: sample the vehicle's flanks along it.
        steps = max(2, int(station.approach / (grid.resolution / 2.0)))
        for i in range(steps + 1):
            centre = start.advanced(station.approach * i / steps)
            for side in (half_width + docking_gap, -(half_width + docking_gap)):
                x, y = centre.to_parent((0.0, side))
                if grid.value_at(x, y) >= OCCUPIED:
                    problems.append(
                        f"the docking run into station '{station.name}' passes within "
                        f'{docking_gap:.2f} m of an obstacle')
                    break
            else:
                continue
            break
    x_min, y_min, x_max, y_max = grid.bounds
    for zone in site.zones:
        xs = [p[0] for p in zone.polygon]
        ys = [p[1] for p in zone.polygon]
        if max(xs) < x_min or min(xs) > x_max or max(ys) < y_min or min(ys) > y_max:
            problems.append(f"zone '{zone.name}' lies entirely outside the map")
    return problems
