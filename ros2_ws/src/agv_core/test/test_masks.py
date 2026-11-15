"""Keep-out and speed masks, and the checks of a site against its map."""

from agv_core.grid import FREE, GridMap, OCCUPIED
from agv_core.masks import keepout_mask, site_map_problems, speed_mask, SPEED_STEP, speed_value
from agv_core.sites import Site
import numpy as np
import pytest

HALF_LENGTH, HALF_WIDTH = 0.30, 0.22


def hall():
    """Build a 12 x 8 m hall with walls, origin at its south-west corner."""
    data = np.full((160, 240), FREE, dtype=np.int16)
    data[0, :] = data[-1, :] = OCCUPIED
    data[:, 0] = data[:, -1] = OCCUPIED
    return GridMap(data, 0.05, (0.0, 0.0))


def site(stations=None, zones=None):
    return Site.from_dict({
        'name': 'hall',
        'stations': stations or {
            'charger': {'kind': 'charger', 'pose': [6.0, 1.0, -90]},
            'dock': {'kind': 'transfer', 'pose': [10.5, 4.0, 0]},
        },
        'zones': zones if zones is not None else [
            {'name': 'storage', 'kind': 'keepout',
             'polygon': [[2.0, 5.0], [5.0, 5.0], [5.0, 7.0], [2.0, 7.0]]},
            {'name': 'aisle', 'kind': 'speed', 'limit': 0.5,
             'polygon': [[6.0, 5.0], [10.0, 5.0], [10.0, 7.0], [6.0, 7.0]]},
            {'name': 'crossing', 'kind': 'speed', 'limit': 0.25,
             'polygon': [[9.0, 4.0], [11.0, 4.0], [11.0, 6.0], [9.0, 6.0]]},
        ],
    })


def test_the_keepout_mask_is_occupied_inside_the_zones_only():
    grid = hall()
    mask = keepout_mask(site(), grid)
    assert mask.data.shape == grid.data.shape and mask.resolution == grid.resolution
    assert mask.origin == grid.origin
    assert mask.value_at(3.5, 6.0) == OCCUPIED
    assert mask.value_at(2.03, 5.03) == OCCUPIED and mask.value_at(4.97, 6.97) == OCCUPIED
    assert mask.value_at(1.97, 6.0) == FREE and mask.value_at(3.5, 4.97) == FREE
    # The walls of the map are not zones: the mask only knows the site file.
    assert mask.value_at(0.02, 0.02) == FREE
    assert mask.occupied.sum() == 60 * 40


def test_the_speed_mask_holds_limits_in_centimetres_per_second():
    mask = speed_mask(site(), hall())
    assert mask.value_at(7.0, 6.5) == 50
    assert mask.value_at(10.5, 4.5) == 25
    assert mask.value_at(1.0, 1.0) == 0                 # no limit
    assert mask.value_at(3.5, 6.0) == 0                 # keep-out zones are not speed zones


def test_where_speed_zones_overlap_the_lower_limit_wins():
    mask = speed_mask(site(), hall())
    assert mask.value_at(9.5, 5.5) == 25
    # Whichever order the zones are written in.
    zones = site().zones
    reordered = Site('hall', 'map', site().stations, zones[::-1])
    assert speed_mask(reordered, hall()).value_at(9.5, 5.5) == 25


@pytest.mark.parametrize(
    'limit, value', [(0.5, 50), (0.254, 25), (1.0, 100), (2.0, 100), (0.001, 1)])
def test_speed_values_stay_between_one_and_a_hundred(limit, value):
    assert speed_value(limit) == value
    assert SPEED_STEP == 0.01


def test_a_site_without_zones_gives_empty_masks():
    empty = site(zones=[])
    assert keepout_mask(empty, hall()).occupied.sum() == 0
    assert speed_mask(empty, hall()).data.max() == 0


def test_a_good_site_has_no_problems_on_its_map():
    assert site_map_problems(site(), hall(), HALF_LENGTH, HALF_WIDTH) == []


def test_a_station_in_a_wall_or_off_the_map_is_a_problem():
    in_wall = site(stations={'dock': {'kind': 'transfer', 'pose': [11.98, 4.0, 0]}})
    assert site_map_problems(in_wall, hall(), HALF_LENGTH, HALF_WIDTH) == [
        "station 'dock' is not on free, mapped floor"]
    outside = site(stations={'dock': {'kind': 'transfer', 'pose': [20.0, 4.0, 0]}})
    assert "station 'dock' is not on free, mapped floor" in site_map_problems(
        outside, hall(), HALF_LENGTH, HALF_WIDTH)


def test_an_approach_pose_off_the_floor_is_a_problem():
    # Docked at x = 0.5 heading east: the approach pose is 0.8 m back, behind the west wall.
    backwards = site(stations={'dock': {'kind': 'transfer', 'pose': [0.5, 4.0, 0]}})
    problems = site_map_problems(backwards, hall(), HALF_LENGTH, HALF_WIDTH)
    assert problems == ["the approach pose of station 'dock' is not on free, mapped floor"]


def test_the_approach_pose_needs_room_to_turn_on_the_spot():
    grid = hall()
    # The approach pose is (9.7, 4.0), facing east. A crate 0.30 m behind it is clear of
    # the docking run, and inside the 0.37 m circle the corners sweep in a turn.
    grid.fill_polygon([(9.0, 3.8), (9.4, 3.8), (9.4, 4.2), (9.0, 4.2)], OCCUPIED)
    problems = site_map_problems(site(), grid, HALF_LENGTH, HALF_WIDTH)
    assert len(problems) == 1
    assert "approach pose of station 'dock' is 0.30 m from the nearest obstacle" in problems[0]
    assert 'needs 0.42 m' in problems[0]


def test_the_docking_run_needs_its_sides_free():
    grid = hall()
    # A post beside the run, 0.28 m from its centre line: clear of the vehicle (0.22), inside
    # the 0.10 m the run keeps free on each side.
    grid.fill_polygon([(10.1, 4.27), (10.2, 4.27), (10.2, 4.37), (10.1, 4.37)], OCCUPIED)
    problems = site_map_problems(site(), grid, HALF_LENGTH, HALF_WIDTH)
    assert problems == ["the docking run into station 'dock' passes within 0.10 m of an obstacle"]
    assert site_map_problems(site(), grid, HALF_LENGTH, HALF_WIDTH, docking_gap=0.0) == []


def test_a_zone_that_misses_the_map_is_a_problem():
    stray = site(zones=[{'name': 'elsewhere', 'kind': 'keepout',
                         'polygon': [[30.0, 30.0], [32.0, 30.0], [32.0, 32.0]]}])
    assert site_map_problems(stray, hall(), HALF_LENGTH, HALF_WIDTH) == [
        "zone 'elsewhere' lies entirely outside the map"]
