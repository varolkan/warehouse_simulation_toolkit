"""Poses, angles and polygons."""

import math

from agv_core.geometry import (
    angle_diff, normalize_angle, path_length, point_in_polygon, point_polygon_distance,
    point_segment_distance, polygon_area, polygon_is_simple, Pose2D, rectangle,
    segments_intersect)
import pytest

SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]


@pytest.mark.parametrize('angle, expected', [
    (0.0, 0.0), (math.pi, math.pi), (-math.pi, math.pi), (3 * math.pi, math.pi),
    (2 * math.pi, 0.0), (-2 * math.pi, 0.0), (math.pi + 0.1, -math.pi + 0.1),
    (-math.pi - 0.1, math.pi - 0.1), (7.0, 7.0 - 2 * math.pi),
])
def test_angles_wrap_into_minus_pi_to_pi(angle, expected):
    assert normalize_angle(angle) == pytest.approx(expected)


def test_angle_diff_takes_the_short_way_round():
    assert angle_diff(math.radians(170), math.radians(-170)) == pytest.approx(math.radians(-20))
    assert angle_diff(math.radians(-170), math.radians(170)) == pytest.approx(math.radians(20))
    assert angle_diff(0.3, 0.1) == pytest.approx(0.2)


def test_advancing_a_pose_moves_along_its_heading():
    pose = Pose2D(1.0, 2.0, math.pi / 2)
    ahead = pose.advanced(0.5)
    assert (ahead.x, ahead.y, ahead.yaw) == pytest.approx((1.0, 2.5, math.pi / 2))
    back_and_left = pose.advanced(-1.0, left=0.25)
    assert (back_and_left.x, back_and_left.y) == pytest.approx((0.75, 1.0))


def test_local_and_parent_frames_are_inverse():
    pose = Pose2D(-3.0, 4.0, 2.1)
    for point in [(0.0, 0.0), (1.5, -2.0), (-7.0, 0.3)]:
        assert pose.to_local(pose.to_parent(point)) == pytest.approx(point)
        assert pose.to_parent(pose.to_local(point)) == pytest.approx(point)
    # One metre ahead of the pose is (1, 0) in its own frame.
    assert pose.to_local((pose.advanced(1.0).x, pose.advanced(1.0).y)) == pytest.approx((1.0, 0.0))


def test_relative_pose():
    origin = Pose2D(1.0, 1.0, math.pi / 2)
    seen = Pose2D(1.0, 3.0, math.pi).relative_to(origin)
    assert (seen.x, seen.y, seen.yaw) == pytest.approx((2.0, 0.0, math.pi / 2))


def test_a_pose_with_a_nan_is_not_finite():
    assert Pose2D(1.0, 2.0, 0.5).is_finite()
    assert not Pose2D(float('nan'), 0.0, 0.0).is_finite()
    assert not Pose2D(0.0, float('inf'), 0.0).is_finite()


def test_polygon_area_is_signed():
    assert polygon_area(SQUARE) == pytest.approx(4.0)
    assert polygon_area(SQUARE[::-1]) == pytest.approx(-4.0)


def test_point_in_polygon_counts_the_edge_as_inside():
    assert point_in_polygon((1.0, 1.0), SQUARE)
    assert point_in_polygon((0.0, 1.0), SQUARE)
    assert point_in_polygon((2.0, 2.0), SQUARE)
    assert not point_in_polygon((2.01, 1.0), SQUARE)
    assert not point_in_polygon((-0.01, -0.01), SQUARE)


def test_point_in_a_concave_polygon():
    ell = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
    assert point_in_polygon((0.5, 2.5), ell)
    assert point_in_polygon((2.5, 0.5), ell)
    assert not point_in_polygon((2.0, 2.0), ell)


def test_distances_to_segments_and_polygons():
    assert point_segment_distance((1.0, 1.0), (0.0, 0.0), (2.0, 0.0)) == pytest.approx(1.0)
    assert point_segment_distance((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)) == pytest.approx(5.0)
    assert point_segment_distance((-1.0, 0.0), (0.0, 0.0), (2.0, 0.0)) == pytest.approx(1.0)
    assert point_polygon_distance((1.0, 1.0), SQUARE) == 0.0
    assert point_polygon_distance((3.0, 1.0), SQUARE) == pytest.approx(1.0)
    assert point_polygon_distance((3.0, 3.0), SQUARE) == pytest.approx(math.sqrt(2.0))


def test_segments_cross_only_when_they_really_cross():
    assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0))
    assert not segments_intersect((0, 0), (1, 1), (2, 2), (3, 3))
    assert not segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))
    # Sharing an end point is not a crossing: neighbouring polygon edges do.
    assert not segments_intersect((0, 0), (1, 0), (1, 0), (1, 1))


def test_a_bow_tie_is_not_a_simple_polygon():
    assert polygon_is_simple(SQUARE)
    assert not polygon_is_simple([(0, 0), (2, 2), (2, 0), (0, 2)])


def test_rectangle_and_path_length():
    corners = rectangle(0.3, 0.2)
    assert len(corners) == 4
    assert polygon_area(corners) == pytest.approx(0.24)
    assert path_length([(0, 0), (3, 0), (3, 4)]) == pytest.approx(7.0)
    assert path_length([(1, 1)]) == 0.0
    assert path_length([]) == 0.0
