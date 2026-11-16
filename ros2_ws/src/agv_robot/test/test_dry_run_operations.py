"""A shift on the stand-in vehicle: refusals, e-stop, obstacle, balky load handler, cancel.

One launch, the tests in order (unittest runs them alphabetically, hence the numbers), the
vehicle carrying on from where the last left it. The load handler refuses its first
command, so the first loading also exercises the mission's retry.
"""

import math
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from dry_run_support import (  # noqa: E402
    ABORTED, CANCELED, Client, dry_run_launch, SITE, SUCCEEDED)
import launch_testing  # noqa: E402
import launch_testing.asserts  # noqa: E402
import pytest  # noqa: E402
import rclpy  # noqa: E402


@pytest.mark.launch_test
def generate_test_description():
    return dry_run_launch(load_rejections=1)


class TestDryRunOperations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('dry_run_operations_test')
        cls.client = Client(cls.node)
        cls.client.wait_until_ready()

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_1_orders_that_cannot_be_carried_out_are_refused_with_the_reason(self):
        client = self.client
        for (pickup, dropoff), expected in (
                (('inbound', 'shipping'),
                 "unknown dropoff 'shipping' (transfer stations here: inbound, rack_a, rack_b, "
                 'packing, outbound, buffer)'),
                (('charger', 'packing'), "'charger' is a charger, not a transfer station"),
                (('packing', 'packing'), "pickup and dropoff are both 'packing'"),
                (('', 'packing'), 'the order names no pickup and the vehicle carries no load')):
            response = client.order(pickup, dropoff, timeout=30.0)
            self.assertEqual(response.status, ABORTED)
            self.assertEqual((response.result.outcome, response.result.message),
                             ('refused', expected))
        # None of it moved the vehicle.
        along, across, _ = client.docking_error('charger')
        self.assertLess(math.hypot(along, across), 0.02)

    def test_2_the_estop_holds_an_order_until_it_is_reset(self):
        client = self.client
        handle = client.send('inbound', 'rack_b')
        refused = handle.get_result_async()
        self.assertTrue(client.spin_until(
            lambda: refused.done() or client.feedback and client.feedback[-1].state == 'to_pickup'
            and client.truth is not None and len(client.track) > 2
            and abs(client.track[-1][3]) > 0.3, 120.0))
        self.assertFalse(refused.done(), refused.done() and refused.result().result.message)
        client.press_estop(True)
        self.assertTrue(client.spin_until(
            lambda: client.last_safety() is not None and client.last_safety().state == 'estop',
            5.0))
        pressed_at = client.truth
        client.spin_for(2.0)
        # Stopped within a few centimetres of where it was, and holding.
        self.assertLess(pressed_at.distance_to(client.truth), 0.30)
        stopped_at = client.truth
        client.spin_for(5.0)
        self.assertLess(stopped_at.distance_to(client.truth), 0.005)
        self.assertEqual(client.last_status().holding, 'e-stop active')
        # Releasing the button is not enough; the reset is.
        client.press_estop(False)
        client.spin_for(2.0)
        self.assertEqual(client.last_safety().state, 'estop')
        self.assertLess(stopped_at.distance_to(client.truth), 0.005)
        reply = client.reset_safety()
        self.assertTrue(reply.success, reply.message)
        response = client.wait_result(handle, 400.0)
        self.assertEqual(response.status, SUCCEEDED, response.result.message)
        self.assertEqual(response.result.message, "delivered from 'inbound' to 'rack_b'")
        # The first load command was refused by the handler and sent again.
        self.assertIn('loading', [f.state for f in client.feedback])

    def test_3_something_put_in_the_way_is_not_hit(self):
        client = self.client
        handle = client.send('rack_b', 'outbound')
        self.assertTrue(client.spin_until(
            lambda: client.feedback and client.feedback[-1].state == 'to_dropoff'
            and abs(client.track[-1][3]) > 0.5, 200.0))
        truth = client.truth
        # A crate 0.6 m in front of the bumper.
        crate = truth.advanced(0.30 + 0.6 + 0.2)
        client.put(f'box {crate.x:.3f} {crate.y:.3f} 0.4 0.4 {math.degrees(truth.yaw):.1f}')
        closest = math.inf
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            client.spin_for(0.1)
            local = client.truth.to_local((crate.x, crate.y))
            dx = max(abs(local[0]) - 0.2 - 0.30, 0.0)
            dy = max(abs(local[1]) - 0.2 - 0.22, 0.0)
            closest = min(closest, math.hypot(dx, dy))
        self.assertGreater(closest, 0.03, 'came within 3 cm of the crate')
        slowed = [s for s in client.safety if s.state in ('slow', 'protective_stop')]
        self.assertTrue(slowed, 'the gate never saw the crate')
        client.put('')
        response = client.wait_result(handle, 400.0)
        self.assertEqual(response.status, SUCCEEDED, response.result.message)

    def test_4_a_cancelled_order_brings_the_vehicle_home(self):
        client = self.client
        handle = client.send('outbound', 'inbound')
        self.assertTrue(client.spin_until(
            lambda: client.feedback and client.feedback[-1].state == 'to_dropoff'
            and client.feedback[-1].distance_driven > 3.0, 300.0))
        cancel = handle.cancel_goal_async()
        self.assertTrue(client.spin_until(cancel.done, 10.0))
        response = client.wait_result(handle, 300.0)
        self.assertEqual(response.status, CANCELED)
        self.assertEqual(response.result.message,
                         'cancelled with the load on board, back at the charger')
        self.assertTrue(client.spin_until(
            lambda: client.last_status().charging and client.last_status().load == 'loaded',
            10.0))
        self.assertIsNone(SITE.keepout_at((client.truth.x, client.truth.y)))


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_processes_exited_cleanly(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
