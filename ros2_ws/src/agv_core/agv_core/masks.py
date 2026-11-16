"""Keep-out and speed-limit masks for Nav2, drawn from the site file onto the map's grid.

Nav2's costmap filters read a zone as an image the size of the map. The site file is the
one place zones are written down; these functions turn it into images and check the site
against the map while they are at it. The masks are files in the repository, so the test
suite regenerates them and fails if a zone was edited and the masks were not.

Nav2 applies costmap filters after inflation, so a keep-out cell is never inflated, and a
footprint that touches a lethal cell is stuck: the controller and the back-up recovery both
check the pose the vehicle is in before any motion, and refuse every motion from it. Each
costmap therefore gets its own version of the zones:

Controller mask (local costmap): lethal deeper than CONTROLLER_RIM inside a zone; the
               zone's outer rim is RIM_VALUE, expensive but not lethal. The controller
               stops the vehicle before it reaches the lethal core, and a vehicle that
               grazed a zone (it stopped late, or its position estimate moved) can still
               drive out of it.
Planner mask:  (global costmap) the zones, lethal, with a band round them as wide as the
               vehicle's turning radius at BAND_VALUE, fading to nothing over a further
               PLANNER_FADE. The planner plans for the vehicle's centre; the band keeps
               routes far enough from a zone that the body, turning or cutting a corner,
               stays out of it. Expensive rather than closed, so that a vehicle that has
               strayed into the band can still plan its way out; graded, so that routes
               keep to the middle of a gap between zones and round a zone's corner at a
               distance rather than along the band's edge (the planner's path smoother
               ignores costs short of lethal and cuts corners into a band of one value).
Speed mask:    cell value v means "no faster than v x SPEED_STEP m/s here"; 0 is no limit.
"""

from typing import List

from agv_core.geometry import point_polygon_distance
from agv_core.grid import FREE, GridMap, OCCUPIED
from agv_core.sites import Site

# One grey level of the speed mask, in m/s. nav2.yaml's speed_filter_info_server has the
# same number as its `multiplier`.
SPEED_STEP = 0.01


def keepout_mask(site: Site, grid: GridMap, grow: float = 0.0) -> GridMap:
    """Return a grid like `grid`, occupied inside every keep-out zone, free elsewhere.

    With `grow`, every cell within that distance of a zone is occupied too.
    """
    mask = grid.like(FREE)
    for zone in site.keepout_zones:
        mask.fill_polygon(zone.polygon, OCCUPIED)
    if grow > 0.0:
        mask.data[mask.inflated(grow, unknown_is_blocked=False)] = OCCUPIED
    return mask


# Mask values short of lethal. Nav2's keep-out filter turns 0..100 into a cost in
# proportion, so 90 is a cost of 229 of 254: under the 253 at which the planner stops and
# the 254 at which the controller and the recoveries see a collision.
BAND_VALUE = 90
RIM_VALUE = BAND_VALUE
# How far into a zone the controller lets the vehicle's outline go before it is a
# collision: two cells. The local costmap is drawn in the odometry frame, so a zone's edge
# lands up to a cell either side of where it is on the map, and a footprint drawn on the
# grid reaches up to half a cell beyond the outline.
CONTROLLER_RIM = 0.10
# Beyond the planner's band, the cost fades to nothing over this distance.
PLANNER_FADE = 0.25


def controller_keepout_mask(site: Site, grid: GridMap, rim: float = CONTROLLER_RIM) -> GridMap:
    """Return the zones for the controller: lethal deeper than `rim` inside, RIM_VALUE in the rim.

    A zone thinner than twice the rim has no lethal core and is all rim.
    """
    mask = keepout_mask(site, grid)
    outside = grid.like(OCCUPIED)
    outside.data[mask.occupied] = FREE
    rim_cells = mask.occupied & outside.inflated(rim, unknown_is_blocked=False)
    mask.data[rim_cells] = RIM_VALUE
    return mask


def planner_keepout_mask(site: Site, grid: GridMap, band: float,
                         fade: float = PLANNER_FADE) -> GridMap:
    """Return the keep-out zones with a band `band` wide round them, fading out over `fade`.

    Cells within `band` of a zone are BAND_VALUE; beyond, the value falls in even steps to
    nothing at `band` + `fade`.
    """
    zones = keepout_mask(site, grid)
    mask = grid.like(FREE)
    rings = int(round(fade / grid.resolution))
    for ring in range(rings, 0, -1):
        # Outermost first: each nearer ring overwrites the one before it.
        within = zones.inflated(band + ring * grid.resolution, unknown_is_blocked=False)
        mask.data[within] = max(1, int(round(BAND_VALUE * (rings + 1 - ring) / (rings + 1))))
    mask.data[zones.inflated(band, unknown_is_blocked=False)] = BAND_VALUE
    mask.data[zones.occupied] = OCCUPIED
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
        grown = [z.name for z in site.keepout_zones
                 if point_polygon_distance((start.x, start.y), z.polygon) <= half_width]
        if grown:
            problems.append(
                f"the approach pose of station '{station.name}' is within the vehicle's half "
                f"width of keep-out zone '{grown[0]}': navigation cannot plan to it")
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
