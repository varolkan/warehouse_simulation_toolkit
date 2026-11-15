"""Occupancy grids: map files, clearance, ray casting and route planning."""

import math

from agv_core.grid import FREE, GridMap, MapError, OCCUPIED, read_pgm, UNKNOWN, write_pgm
import numpy as np
import pytest


def room(width=10.0, height=6.0, resolution=0.05, origin=(-5.0, -3.0)):
    """Build a free rectangle with a one-cell wall all round."""
    data = np.full((int(round(height / resolution)), int(round(width / resolution))), FREE,
                   dtype=np.int16)
    data[0, :] = data[-1, :] = OCCUPIED
    data[:, 0] = data[:, -1] = OCCUPIED
    return GridMap(data, resolution, origin)


def wall(grid, x, y0, y1):
    """Add a one-cell-thick wall at x from y0 to y1."""
    col, row0 = grid.cell_of(x, y0)
    _, row1 = grid.cell_of(x, y1)
    grid.data[row0:row1 + 1, col] = OCCUPIED
    return grid


def test_cells_and_coordinates():
    grid = room()
    assert (grid.width, grid.height) == (200, 120)
    assert grid.bounds == pytest.approx((-5.0, -3.0, 5.0, 3.0))
    assert grid.cell_of(-5.0, -3.0) == (0, 0)
    assert grid.cell_of(-4.951, -2.951) == (0, 0)
    assert grid.cell_of(-4.949, -2.949) == (1, 1)
    assert grid.centre_of(0, 0) == pytest.approx((-4.975, -2.975))
    assert grid.cell_of(*grid.centre_of(37, 91)) == (37, 91)
    assert grid.contains_cell(199, 119) and not grid.contains_cell(200, 0)
    assert not grid.contains_cell(-1, 5)


def test_values_outside_the_grid_are_unknown():
    grid = room()
    assert grid.value_at(0.0, 0.0) == FREE and grid.is_free(0.0, 0.0)
    assert grid.value_at(-4.99, 0.0) == OCCUPIED
    assert grid.value_at(-5.01, 0.0) == UNKNOWN
    assert grid.value_at(100.0, 100.0) == UNKNOWN
    assert not grid.is_free(100.0, 100.0)


def test_bad_grids_are_refused():
    with pytest.raises(MapError):
        GridMap(np.zeros(5), 0.05)
    with pytest.raises(MapError):
        GridMap(np.zeros((5, 5)), 0.0)


def test_a_map_survives_being_saved_and_loaded(tmp_path):
    grid = room()
    grid.data[10:20, 30:40] = UNKNOWN
    grid.data[50, 60] = OCCUPIED
    path = str(tmp_path / 'hall.yaml')
    grid.save(path)
    assert (tmp_path / 'hall.pgm').exists()
    loaded = GridMap.load(path)
    assert loaded.resolution == pytest.approx(0.05)
    assert loaded.origin == pytest.approx((-5.0, -3.0))
    assert np.array_equal(loaded.data, grid.data)
    # Row 0 of the array is the bottom of the map; row 0 of the image is its top.
    pixels = read_pgm(str(tmp_path / 'hall.pgm'))
    assert pixels.shape == (120, 200)
    assert pixels[119 - 50, 60] == 0 and pixels[0, 0] == 0
    assert pixels[60, 100] == 254


def test_a_raw_map_keeps_its_values(tmp_path):
    grid = room().like(0)
    grid.data[10:20, 10:20] = 40
    grid.data[30:40, 30:40] = 100
    path = str(tmp_path / 'speed.yaml')
    grid.save(path, mode='raw')
    assert 'mode: raw' in (tmp_path / 'speed.yaml').read_text()
    loaded = GridMap.load(path)
    assert np.array_equal(loaded.data, grid.data)
    with pytest.raises(MapError):
        grid.save(path, mode='fancy')


def test_loading_thresholds_grey_levels_like_map_server(tmp_path):
    # 0 black: occupied. 254 white: free. 205 map_saver's grey: unknown.
    pixels = np.array([[0, 254, 205], [100, 210, 80]], dtype=np.uint8)
    write_pgm(str(tmp_path / 'm.pgm'), pixels)
    (tmp_path / 'm.yaml').write_text(
        'image: m.pgm\nresolution: 0.1\norigin: [1.0, 2.0, 0.0]\nnegate: 0\n'
        'occupied_thresh: 0.65\nfree_thresh: 0.196\n')
    grid = GridMap.load(str(tmp_path / 'm.yaml'))
    # The file's second row is the bottom row of the map.
    assert grid.data.tolist() == [[UNKNOWN, FREE, OCCUPIED], [OCCUPIED, FREE, UNKNOWN]]
    assert grid.origin == (1.0, 2.0)
    (tmp_path / 'n.yaml').write_text(
        'image: m.pgm\nresolution: 0.1\norigin: [0, 0, 0]\nnegate: 1\n')
    negated = GridMap.load(str(tmp_path / 'n.yaml'))
    assert negated.data[1].tolist() == [FREE, OCCUPIED, OCCUPIED]


def test_a_free_threshold_of_a_quarter_turns_unmapped_cells_into_floor(tmp_path):
    """Why map files here say free_thresh 0.196 and not the 0.25 that is often written."""
    write_pgm(str(tmp_path / 'm.pgm'), np.array([[205]], dtype=np.uint8))
    for free_thresh, expected in ((0.196, UNKNOWN), (0.25, FREE)):
        (tmp_path / 'm.yaml').write_text(
            f'image: m.pgm\nresolution: 0.05\norigin: [0, 0, 0]\nnegate: 0\n'
            f'occupied_thresh: 0.65\nfree_thresh: {free_thresh}\n')
        assert GridMap.load(str(tmp_path / 'm.yaml')).data[0, 0] == expected
    saved = GridMap(np.array([[UNKNOWN]], dtype=np.int16), 0.05)
    saved.save(str(tmp_path / 's.yaml'))
    assert 'free_thresh: 0.196' in (tmp_path / 's.yaml').read_text()
    assert GridMap.load(str(tmp_path / 's.yaml')).data[0, 0] == UNKNOWN


def test_plain_and_binary_pgm_are_both_read(tmp_path):
    (tmp_path / 'plain.pgm').write_text('P2\n# a comment\n3 2\n255\n0 127 255\n10 20 30\n')
    assert read_pgm(str(tmp_path / 'plain.pgm')).tolist() == [[0, 127, 255], [10, 20, 30]]
    (tmp_path / 'deep.pgm').write_bytes(b'P5\n2 1\n65535\n' + bytes([0xFF, 0xFF, 0x00, 0x00]))
    assert read_pgm(str(tmp_path / 'deep.pgm')).tolist() == [[255, 0]]


@pytest.mark.parametrize('content, expected', [
    (b'P6\n2 2\n255\n' + bytes(12), 'not a PGM'),
    (b'P5\n4 4\n255\n' + bytes(3), 'shorter than'),
    (b'P5\n4', 'truncated'),
    (b'P5\nfour 4\n255\n', 'bad PGM header'),
])
def test_broken_images_are_refused(tmp_path, content, expected):
    path = tmp_path / 'broken.pgm'
    path.write_bytes(content)
    with pytest.raises(MapError) as caught:
        read_pgm(str(path))
    assert expected in str(caught.value)


def test_broken_map_descriptions_are_refused(tmp_path):
    with pytest.raises(MapError):
        GridMap.load(str(tmp_path / 'missing.yaml'))
    (tmp_path / 'a.yaml').write_text('resolution: 0.05\norigin: [0, 0, 0]\n')
    with pytest.raises(MapError) as caught:
        GridMap.load(str(tmp_path / 'a.yaml'))
    assert "'image' is missing" in str(caught.value)
    (tmp_path / 'b.yaml').write_text('image: gone.pgm\nresolution: 0.05\norigin: [0, 0, 0]\n')
    with pytest.raises(MapError):
        GridMap.load(str(tmp_path / 'b.yaml'))
    write_pgm(str(tmp_path / 'c.pgm'), np.zeros((2, 2), dtype=np.uint8))
    (tmp_path / 'c.yaml').write_text('image: c.pgm\nresolution: 0.05\norigin: [0, 0, 0.5]\n')
    with pytest.raises(MapError) as caught:
        GridMap.load(str(tmp_path / 'c.yaml'))
    assert 'rotated' in str(caught.value)
    (tmp_path / 'd.yaml').write_text('- just\n- a list\n')
    with pytest.raises(MapError):
        GridMap.load(str(tmp_path / 'd.yaml'))


def test_inflation_grows_obstacles_by_a_radius():
    grid = GridMap(np.zeros((41, 41), dtype=np.int16), 0.05, (-1.025, -1.025))
    grid.data[20, 20] = OCCUPIED
    inflated = grid.inflated(0.30, unknown_is_blocked=False)
    assert inflated[20, 20]
    assert inflated[20, 26] and not inflated[20, 27]      # 6 cells = 0.30 m, 7 is beyond
    assert inflated[24, 24] and not inflated[25, 25]      # 5.66 cells yes, 7.07 no
    assert inflated.sum() == pytest.approx(math.pi * 36, rel=0.08)
    assert np.array_equal(grid.inflated(0.0, unknown_is_blocked=False), grid.occupied)


def test_unknown_space_blocks_unless_told_otherwise():
    grid = GridMap(np.zeros((10, 10), dtype=np.int16), 0.1)
    grid.data[5, 5] = UNKNOWN
    assert grid.blocked()[5, 5]
    assert not grid.blocked(unknown_is_blocked=False)[5, 5]


def test_clearance_is_measured_to_the_surface_of_the_nearest_obstacle():
    grid = room()
    # The west wall's cell spans x -5.00 .. -4.95.
    assert grid.clearance_at(-4.0, 0.0) == pytest.approx(0.95, abs=1e-6)
    assert grid.clearance_at(0.0, 2.0) == pytest.approx(0.95, abs=1e-6)
    assert grid.clearance_at(0.0, 0.0, search=1.5) == 1.5      # nothing within reach
    assert grid.clearance_at(-4.97, 0.0) == 0.0                # inside the wall
    assert grid.clearance_at(50.0, 50.0) == 3.0                # off the map


def test_filling_a_polygon_sets_the_cells_whose_centre_is_inside():
    grid = GridMap(np.zeros((20, 20), dtype=np.int16), 0.1)
    count = grid.fill_polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.0), (0.5, 1.0)], OCCUPIED)
    assert count == 50                                          # 10 x 5 cells
    assert grid.value_at(0.55, 0.55) == OCCUPIED and grid.value_at(1.45, 0.95) == OCCUPIED
    assert grid.value_at(0.45, 0.55) == FREE and grid.value_at(1.0, 1.05) == FREE
    triangle = grid.like()
    triangle.fill_polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)], OCCUPIED)
    assert triangle.value_at(0.25, 0.25) == OCCUPIED and triangle.value_at(1.5, 1.5) == FREE
    assert triangle.occupied.sum() == pytest.approx(200, abs=12)


def test_rays_stop_at_the_first_wall():
    grid = room()                      # walls at x = +-5 (inner faces at +-4.95), y = +-3
    angles = np.radians([0.0, 90.0, 180.0, 270.0, 45.0])
    ranges = grid.ray_cast(0.0, 0.0, angles, max_range=12.0)
    step = grid.resolution / 2.0
    for measured, true in zip(ranges, (4.95, 2.95, 4.95, 2.95, 2.95 * math.sqrt(2.0))):
        # Sampled every half cell: a hit is reported within one step of the surface.
        assert true - 1e-9 <= measured <= true + 2 * step
    off_centre = grid.ray_cast(3.0, 1.0, np.radians([0.0, 180.0]), max_range=12.0)
    assert off_centre[0] == pytest.approx(1.95, abs=2 * step)
    assert off_centre[1] == pytest.approx(7.95, abs=2 * step)


def test_rays_that_hit_nothing_report_infinity():
    grid = room()
    assert np.isinf(grid.ray_cast(0.0, 0.0, np.array([0.0]), max_range=3.0)[0])
    open_field = GridMap(np.zeros((50, 50), dtype=np.int16), 0.1)
    assert np.all(np.isinf(open_field.ray_cast(2.5, 2.5, np.linspace(0, 6.2, 16), 30.0)))
    # Unknown cells are see-through, not walls.
    grid.data[60, 110:120] = UNKNOWN
    assert grid.ray_cast(0.0, 0.0, np.array([0.0]), 12.0)[0] == pytest.approx(4.95, abs=0.05)


def test_a_ray_sees_a_thin_obstacle_between_it_and_the_wall():
    grid = wall(room(), 2.0, -0.5, 0.5)
    ranges = grid.ray_cast(0.0, 0.0, np.radians([0.0, 30.0]), 12.0)
    assert ranges[0] == pytest.approx(2.0, abs=0.05)
    assert ranges[1] > 4.0                    # passes above the short wall


def test_line_is_clear():
    grid = wall(room(), 0.0, -3.0, 1.0)
    blocked = grid.blocked()
    assert grid.line_is_clear((-2.0, 2.0), (2.0, 2.0), blocked)
    assert not grid.line_is_clear((-2.0, 0.0), (2.0, 0.0), blocked)
    assert not grid.line_is_clear((-2.0, 0.0), (20.0, 0.0), blocked)   # leaves the map


def test_a_route_goes_round_a_wall():
    grid = wall(room(), 0.0, -3.0, 1.0)          # gap between y = 1.05 and the north wall
    route = grid.plan((-3.0, -2.0), (3.0, -2.0), clearance=0.35)
    assert route is not None
    assert route[0] == (-3.0, -2.0) and route[-1] == (3.0, -2.0)
    # Through the gap, and clear of the end of the wall.
    assert max(y for _, y in route) > 1.05 + 0.3
    blocked = grid.inflated(0.35)
    for a, b in zip(route, route[1:]):
        assert grid.line_is_clear(a, b, blocked)
    # The straight line is 6 m; round the wall is about 9.3 m.
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(route, route[1:]))
    assert 8.5 < length < 10.5
    assert len(route) <= 6                                   # corners only, not every cell


def test_a_straight_route_is_two_points():
    route = room().plan((-3.0, 0.0), (3.0, 1.0), clearance=0.3)
    assert route == [(-3.0, 0.0), (3.0, 1.0)]


def test_there_is_no_route_through_a_closed_wall():
    grid = wall(room(), 0.0, -3.0, 3.0)
    assert grid.plan((-3.0, 0.0), (3.0, 0.0), clearance=0.3) is None


def test_a_gap_narrower_than_the_clearance_is_not_a_route():
    grid = wall(wall(room(), 0.0, -3.0, -0.4), 0.0, 0.4, 3.0)     # a 0.75 m gap at y = 0
    assert grid.plan((-3.0, 0.0), (3.0, 0.0), clearance=0.30) is not None
    assert grid.plan((-3.0, 0.0), (3.0, 0.0), clearance=0.45) is None


def test_unknown_space_is_not_driven_through_unless_allowed():
    grid = room()
    col, _ = grid.cell_of(0.0, 0.0)
    grid.data[1:-1, col - 2:col + 3] = UNKNOWN            # an unmapped strip across the room
    assert grid.plan((-3.0, 0.0), (3.0, 0.0), clearance=0.3) is None
    assert grid.plan((-3.0, 0.0), (3.0, 0.0), clearance=0.3, unknown_is_blocked=False) is not None


def test_extra_blocked_cells_are_avoided():
    grid = room()
    zone = grid.polygon_mask([(-0.5, -3.0), (0.5, -3.0), (0.5, 1.5), (-0.5, 1.5)])
    route = grid.plan((-3.0, 0.0), (3.0, 0.0), clearance=0.3, extra_blocked=zone)
    assert route is not None and max(y for _, y in route) > 1.5
    assert grid.plan((-3.0, 0.0), (3.0, 0.0), clearance=0.3) == [(-3.0, 0.0), (3.0, 0.0)]


def test_a_vehicle_docked_close_to_a_wall_can_still_plan_away_from_it():
    grid = room()
    # 0.2 m from the west wall: inside the 0.35 m clearance band, as at a station.
    route = grid.plan((-4.75, 0.0), (3.0, 0.0), clearance=0.35)
    assert route is not None and route[0] == (-4.75, 0.0) and route[-1] == (3.0, 0.0)


def test_points_off_the_map_have_no_route():
    grid = room()
    assert grid.plan((-30.0, 0.0), (3.0, 0.0), clearance=0.3) is None
    assert grid.plan((0.0, 0.0), (0.0, 40.0), clearance=0.3) is None
