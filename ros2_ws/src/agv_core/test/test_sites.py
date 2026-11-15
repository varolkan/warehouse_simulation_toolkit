"""The site file: what loads, what is refused, and what the refusal says."""

import copy
import math

from agv_core.geometry import Pose2D
from agv_core.sites import Site, SiteError
import pytest

GOOD = {
    'name': 'hall',
    'frame': 'map',
    'stations': {
        'charger': {'kind': 'charger', 'pose': [8.0, -8.0, -90]},
        'inbound': {'kind': 'transfer', 'pose': [-8.0, -1.0, 180]},
        'packing': {'kind': 'transfer', 'pose': [5.0, 5.0, 90], 'approach': 1.2},
    },
    'zones': [
        {'name': 'racks', 'kind': 'keepout',
         'polygon': [[2.0, -9.0], [3.0, -9.0], [3.0, -5.0], [2.0, -5.0]]},
        {'name': 'slow', 'kind': 'speed', 'limit': 0.5,
         'polygon': [[4.0, 3.0], [6.0, 3.0], [6.0, 6.0], [4.0, 6.0]]},
        {'name': 'slower', 'kind': 'speed', 'limit': 0.3,
         'polygon': [[5.0, 4.0], [7.0, 4.0], [7.0, 7.0], [5.0, 7.0]]},
    ],
}


def site_with(change):
    data = copy.deepcopy(GOOD)
    change(data)
    return data


def problems_of(data):
    with pytest.raises(SiteError) as caught:
        Site.from_dict(data, source='test.yaml')
    return caught.value.problems


def test_a_good_site_loads():
    site = Site.from_dict(GOOD)
    assert site.name == 'hall' and site.frame == 'map'
    assert list(site.stations) == ['charger', 'inbound', 'packing']
    assert [s.name for s in site.transfer_stations] == ['inbound', 'packing']
    assert [s.name for s in site.chargers] == ['charger']
    assert [z.name for z in site.keepout_zones] == ['racks']
    assert [z.name for z in site.speed_zones] == ['slow', 'slower']


def test_headings_are_given_in_degrees_and_held_in_radians():
    site = Site.from_dict(GOOD)
    assert site.station('charger').pose.yaw == pytest.approx(-math.pi / 2)
    assert site.station('inbound').pose.yaw == pytest.approx(math.pi)


def test_the_approach_pose_is_straight_back_from_the_station():
    site = Site.from_dict(GOOD)
    charger = site.station('charger')          # facing south, default approach
    assert charger.approach == 0.8
    assert (charger.approach_pose.x, charger.approach_pose.y) == pytest.approx((8.0, -7.2))
    assert charger.approach_pose.yaw == pytest.approx(charger.pose.yaw)
    packing = site.station('packing')          # facing north, 1.2 m approach
    assert (packing.approach_pose.x, packing.approach_pose.y) == pytest.approx((5.0, 3.8))


def test_an_unknown_station_is_named_together_with_the_known_ones():
    site = Site.from_dict(GOOD)
    with pytest.raises(KeyError) as caught:
        site.station('shipping')
    message = str(caught.value)
    assert 'shipping' in message
    for known in ('charger', 'inbound', 'packing'):
        assert known in message
    assert site.has_station('inbound') and not site.has_station('shipping')


def test_zone_lookups():
    site = Site.from_dict(GOOD)
    assert site.keepout_at((2.5, -7.0)).name == 'racks'
    assert site.keepout_at((0.0, 0.0)) is None
    assert site.speed_limit_at((4.5, 3.5)) == 0.5
    assert site.speed_limit_at((6.5, 6.5)) == 0.3
    # Where two speed zones overlap the lower limit applies.
    assert site.speed_limit_at((5.5, 5.0)) == 0.3
    assert site.speed_limit_at((0.0, 0.0)) is None


def test_nearest_charger_and_station_at():
    data = site_with(lambda d: d['stations'].update(
        {'charger_west': {'kind': 'charger', 'pose': [-8.0, 6.0, 180]}}))
    site = Site.from_dict(data)
    assert site.nearest_charger(Pose2D(-5.0, 5.0, 0.0)).name == 'charger_west'
    assert site.nearest_charger(Pose2D(6.0, -6.0, 0.0)).name == 'charger'
    assert site.station_at(Pose2D(8.1, -8.05, 0.0)).name == 'charger'
    assert site.station_at(Pose2D(0.0, 0.0, 0.0)) is None
    no_charger = site_with(lambda d: d['stations'].pop('charger'))
    assert Site.from_dict(no_charger).nearest_charger(Pose2D()) is None


def test_zones_are_optional():
    site = Site.from_dict(site_with(lambda d: d.pop('zones')))
    assert site.zones == ()
    assert Site.from_dict(site_with(lambda d: d.update(zones=None))).zones == ()


@pytest.mark.parametrize('change, expected', [
    (lambda d: d.pop('name'), "'name' is missing"),
    (lambda d: d.update(name='  '), "'name' is missing"),
    (lambda d: d.update(frame=7), "'frame' must be"),
    (lambda d: d.update(depot=[0, 0]), 'unknown top-level key(s): depot'),
    (lambda d: d.pop('stations'), "'stations' must be a mapping"),
    (lambda d: d.update(stations={}), "'stations' must be a mapping"),
    (lambda d: d.update(stations=['a']), "'stations' must be a mapping"),
    (lambda d: d['stations'].update({'two words': {'kind': 'transfer', 'pose': [0, 0, 0]}}),
     'one word without spaces'),
    (lambda d: d['stations'].update(dock='here'), "station 'dock': must be a mapping"),
    (lambda d: d['stations']['inbound'].update(kind='depot'), "kind is 'depot'"),
    (lambda d: d['stations']['inbound'].pop('kind'), "kind is 'None'"),
    (lambda d: d['stations']['inbound'].update(pose=[1.0, 2.0]), 'pose must be [x, y, heading'),
    (lambda d: d['stations']['inbound'].update(pose=[1.0, 'north', 0]), 'pose must be'),
    (lambda d: d['stations']['inbound'].update(pose=[1.0, float('nan'), 0]), 'pose must be'),
    (lambda d: d['stations']['inbound'].update(pose=[1.0, True, 0]), 'pose must be'),
    (lambda d: d['stations']['inbound'].update(approach=0.1), 'approach must be between'),
    (lambda d: d['stations']['inbound'].update(approach=5.0), 'approach must be between'),
    (lambda d: d['stations']['inbound'].update(approach='far'), 'approach must be between'),
    (lambda d: d['stations']['inbound'].update(colour='red'), 'unknown key(s): colour'),
    (lambda d: d.update(zones={'a': 1}), "'zones' must be a list"),
    (lambda d: d['zones'].append('fence'), 'must be a mapping with name, kind and polygon'),
    (lambda d: d['zones'][0].pop('name'), 'needs a name'),
    (lambda d: d['zones'][1].update(name='racks'), 'the name is used twice'),
    (lambda d: d['zones'][0].update(kind='danger'), "kind is 'danger'"),
    (lambda d: d['zones'][0].update(polygon=[[0, 0], [1, 1]]), 'at least three'),
    (lambda d: d['zones'][0].update(polygon=[[0, 0], [1, 1], [2]]), 'at least three'),
    (lambda d: d['zones'][0].update(polygon=[[0, 0], [1, 1], [2, 'x']]), 'at least three'),
    (lambda d: d['zones'][0].update(polygon=[[0, 0], [2, 2], [2, 0], [0, 2]]), 'crosses itself'),
    (lambda d: d['zones'][0].update(polygon=[[0, 0], [1, 0], [2, 0]]), 'encloses no area'),
    (lambda d: d['zones'][1].pop('limit'), 'a speed zone needs a limit'),
    (lambda d: d['zones'][1].update(limit=0), 'a speed zone needs a limit'),
    (lambda d: d['zones'][1].update(limit=-0.5), 'a speed zone needs a limit'),
    (lambda d: d['zones'][1].update(limit=9.0), 'a speed zone needs a limit'),
    (lambda d: d['zones'][0].update(limit=0.5), 'a keep-out zone has no limit'),
    (lambda d: d['zones'][0].update(height=2), 'unknown key(s): height'),
    (lambda d: d['stations']['inbound'].update(pose=[2.5, -7.0, 0]),
     "station 'inbound' is inside keep-out zone 'racks'"),
    (lambda d: d['stations']['inbound'].update(pose=[3.3, -7.0, 0]),
     "the docking run into station 'inbound' (0.8 m) crosses keep-out zone 'racks'"),
    (lambda d: d['stations']['inbound'].update(pose=[8.0, -8.3, 0]),
     "stations 'charger' and 'inbound' are 0.30 m apart"),
])
def test_what_a_site_file_is_refused_for(change, expected):
    problems = problems_of(site_with(change))
    assert any(expected in problem for problem in problems), problems


def test_every_problem_is_reported_at_once():
    def break_three_things(data):
        data['name'] = ''
        data['stations']['inbound']['kind'] = 'depot'
        data['zones'][1].pop('limit')

    with pytest.raises(SiteError) as caught:
        Site.from_dict(site_with(break_three_things), source='hall.yaml')
    assert len(caught.value.problems) == 3
    text = str(caught.value)
    assert text.startswith('hall.yaml: 3 problem(s)')
    assert 'depot' in text and 'limit' in text and "'name'" in text


def test_a_site_that_is_not_a_mapping_is_refused():
    assert 'must be a mapping' in problems_of(['stations'])[0]
    assert 'must be a mapping' in problems_of(None)[0]


def test_a_docking_run_that_only_passes_a_zone_is_fine():
    # 0.2 m clear of the zone's edge all the way in.
    data = site_with(lambda d: d['stations']['inbound'].update(pose=[4.0, -7.0, 180]))
    assert Site.from_dict(data).station('inbound').approach_pose.x == pytest.approx(4.8)


def test_the_docked_vehicle_must_not_reach_into_a_keepout_zone():
    # Docked 0.25 m from the zone and facing it: the station and the run into it are
    # outside, the front of a 0.6 m long vehicle is not.
    data = site_with(lambda d: d['stations']['inbound'].update(pose=[3.25, -7.0, 180]))
    site = Site.from_dict(data)
    problems = site.vehicle_problems(half_length=0.30, half_width=0.22)
    assert problems == ["docked at 'inbound' the vehicle reaches into keep-out zone 'racks'"]
    assert Site.from_dict(GOOD).vehicle_problems(0.30, 0.22) == []


def test_loading_from_a_file(tmp_path):
    import yaml

    path = tmp_path / 'site.yaml'
    path.write_text(yaml.safe_dump(GOOD))
    site = Site.load(str(path))
    assert site.source == str(path)
    assert site.station('packing').approach == 1.2


def test_a_missing_or_broken_file_is_a_site_error(tmp_path):
    with pytest.raises(SiteError) as caught:
        Site.load(str(tmp_path / 'nowhere.yaml'))
    assert 'cannot read the file' in str(caught.value)
    broken = tmp_path / 'broken.yaml'
    broken.write_text('stations: [unclosed')
    with pytest.raises(SiteError) as caught:
        Site.load(str(broken))
    assert 'not valid YAML' in str(caught.value)
    empty = tmp_path / 'empty.yaml'
    empty.write_text('')
    with pytest.raises(SiteError):
        Site.load(str(empty))
