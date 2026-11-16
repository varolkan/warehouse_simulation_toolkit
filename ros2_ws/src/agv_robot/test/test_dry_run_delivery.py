"""A whole delivery through the ROS graph, on the stand-in vehicle.

Exercises what the unit tests cannot: AMCL localising on the site's map from the charger,
Nav2 planning and following through the keep-out and speed masks, the gate between every
command and the drive, the station field set, the load handler's service and state, and
the action's lifecycle. Pass and fail are measured against where the vehicle really is.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from dry_run_support import Client, dry_run_launch, SITE, SUCCEEDED  # noqa: E402
import launch_testing  # noqa: E402
import launch_testing.asserts  # noqa: E402
import pytest  # noqa: E402
import rclpy  # noqa: E402


@pytest.mark.launch_test
def generate_test_description():
    return dry_run_launch()


class TestDryRunDelivery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('dry_run_delivery_test')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_delivery_from_goods_in_to_packing_and_back_to_charge(self):
        client = Client(self.node)
        client.wait_until_ready()
        before = len(client.safety)

        response = client.order('inbound', 'packing', timeout=420.0, return_to_charger=True)
        result = response.result
        self.assertEqual(response.status, SUCCEEDED, result.message)
        self.assertTrue(result.success and result.delivered)
        self.assertEqual(result.outcome, 'delivered')
        self.assertEqual(result.message,
                         "delivered from 'inbound' to 'packing', back at the charger")
        self.assertEqual(client.states(), [
            'checking', 'undocking', 'to_pickup', 'docking', 'loading', 'undocking',
            'to_dropoff', 'docking', 'unloading', 'undocking', 'to_charger', 'docking'])
        self.assertGreater(result.distance, 40.0)
        self.assertLess(result.duration, 300.0)

        # Back on the charger: on the contacts, charging, empty. The result can arrive before
        # the status that says so (it is published at 5 Hz): wait for that status.
        self.assertTrue(client.spin_until(
            lambda: client.last_status() is not None and client.last_status().charging
            and client.last_status().state == 'complete', 10.0))
        # Docking is as good as the position AMCL gives it, and against fake_base's scanner,
        # which casts its rays through the 5 cm cells of the map, that is 2 to 6 cm. The
        # stand-in charger takes the vehicle within 8 cm (fake_base_node, charger_contact);
        # precision is measured in Gazebo, against the real geometry.
        along, across, heading = client.docking_error('charger')
        self.assertLess(abs(along), 0.08)
        self.assertLess(abs(across), 0.08)
        self.assertLess(abs(heading), 3.0)
        status = client.last_status()
        self.assertEqual((status.state, status.load), ('complete', 'empty'))

        # Never closer to anything than the gate allows, never in a zone it may not enter.
        for _, x, y, _ in client.track:
            self.assertIsNone(SITE.keepout_at((x, y)), f'in a keep-out zone at ({x}, {y})')
        stops = [s for s in client.safety[before:]
                 if s.state in ('protective_stop', 'estop', 'fault')]
        self.assertEqual(stops, [], f'stopped by the gate: {[s.reason for s in stops][:3]}')

        # Speed limits: no faster than 0.4 m/s once well inside the packing area.
        zone = next(z for z in SITE.speed_zones if z.name == 'packing_area')
        inside = [abs(v) for _, x, y, v in client.track
                  if zone.contains((x, y)) and 3.9 < y < 5.3 and 2.7 < x < 8.3]
        self.assertTrue(inside)
        self.assertLess(max(inside), zone.limit + 0.05)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_processes_exited_cleanly(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
