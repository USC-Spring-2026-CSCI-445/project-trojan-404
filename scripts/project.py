#!/usr/bin/env python3
from typing import Optional, Dict, List
from argparse import ArgumentParser
from math import sqrt, atan2, pi, inf
import math
import json
import numpy as np

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

# Import your existing implementations
from lab8_9_starter import Map, ParticleFilter, angle_to_neg_pi_to_pi  # :contentReference[oaicite:2]{index=2}
from lab10_starter import RrtPlanner, PIDController as WaypointPID, GOAL_THRESHOLD  # :contentReference[oaicite:3]{index=3}


class PFRRTController:
    """
    Combined controller that:
      1) Localizes using a particle filter (by exploring).
      2) Plans with RRT from PF estimate to goal.
      3) Follows that plan with a waypoint PID controller while
         continuing to run the particle filter.
    """

    def __init__(self, pf: ParticleFilter, planner: RrtPlanner, goal_position: Dict[str, float]):
        self._pf = pf
        self._planner = planner
        self.goal_position = goal_position

        # Robot state from odom / laser
        self.current_position: Optional[Dict[str, float]] = None
        self.last_odom: Optional[Dict[str, float]] = None
        self.laserscan: Optional[LaserScan] = None

        # Command publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)

        # PID controllers for tracking waypoints (copied from your ObstacleFreeWaypointController)
        self.linear_pid = WaypointPID(0.3, 0.0, 0.1, 10, -0.22, 0.22)
        self.angular_pid = WaypointPID(0.5, 0.0, 0.2, 10, -2.84, 2.84)

        # Waypoint tracking state
        self.plan: Optional[List[Dict[str, float]]] = None
        self.current_wp_idx: int = 0

        self.rate = rospy.Rate(10)

        # Wait until we have initial odom + scan
        while (self.current_position is None or self.laserscan is None) and (not rospy.is_shutdown()):
            rospy.loginfo("Waiting for /odom and /scan...")
            rospy.sleep(0.1)

    # ----------------------------------------------------------------------
    # Basic callbacks
    # ----------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        new_pose = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        # Use odom delta to propagate PF motion model
        if self.last_odom is not None:
            dx_world = new_pose["x"] - self.last_odom["x"]
            dy_world = new_pose["y"] - self.last_odom["y"]
            dtheta = angle_to_neg_pi_to_pi(new_pose["theta"] - self.last_odom["theta"])

            # convert world delta to robot frame of previous pose
            ct = math.cos(self.last_odom["theta"])
            st = math.sin(self.last_odom["theta"])
            dx_robot = ct * dx_world + st * dy_world
            dy_robot = -st * dx_world + ct * dy_world

            # propagate all particles
            self._pf.move_by(dx_robot, dy_robot, dtheta)

        self.last_odom = new_pose
        self.current_position = new_pose

    def laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    # ----------------------------------------------------------------------
    # Low-level motion primitives
    # ----------------------------------------------------------------------
    def move_forward(self, distance: float):
        """
        Move the robot straight by a commanded distance (meters)
        using a constant velocity profile.
        """
        twist = Twist()
        speed = 0.15  # m/s
        twist.linear.x = speed if distance >= 0 else -speed

        duration = abs(distance) / speed if speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)

    def rotate_in_place(self, angle: float):
        """
        Rotate robot by a relative angle (radians).
        """
        twist = Twist()
        angular_speed = 0.8  # rad/s
        twist.angular.z = angular_speed if angle >= 0.0 else -angular_speed

        duration = abs(angle) / angular_speed if angular_speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ----------------------------------------------------------------------
    # Measurement update
    # ----------------------------------------------------------------------
    def take_measurements(self):
        """
        Use 3 beams (-15°, 0°, +15° in the robot frame) from /scan
        to update the particle filter via its measurement model.
        """
        if self.laserscan is None:
            return

        angle_min = self.laserscan.angle_min
        angle_increment = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        num_ranges = len(ranges)

        mid_idx = num_ranges // 2
        offset = int(15.0 / (angle_increment * 180.0 / math.pi))  # 15 degrees offset

        indices = [max(0, min(num_ranges - 1, mid_idx + i)) for i in (-offset, 0, offset)]
        measurements = []

        for idx in indices:
            z = ranges[idx]
            if z == inf or np.isinf(z):
                if hasattr(self.laserscan, "range_max"):
                    z = self.laserscan.range_max
                else:
                    z = 10.0  # fallback
            angle = angle_min + idx * angle_increment  # angle in robot frame
            measurements.append((z, angle))

        for z, a in measurements:
            self._pf.measure(z, a)

    # ----------------------------------------------------------------------
    # Phase 1: Localization with PF (explore a bit)
    # ----------------------------------------------------------------------
    def localize_with_pf(self, max_steps: int = 400):
        """
        Simple autonomous exploration policy:
          - If front is free, go forward.
          - If obstacle close in front, back up and rotate.
        After each motion, apply PF measurement updates and check convergence.
        """
        
        ######### Your code starts here #########

        def clean_range(r):
            if np.isnan(r) or np.isinf(r):
                return self.laserscan.range_max
            return r

        def get_min_range_between_angles(low_angle, high_angle):
            if self.laserscan is None:
                return self.laserscan.range_max

            angle_min = self.laserscan.angle_min
            angle_inc = self.laserscan.angle_increment
            ranges = self.laserscan.ranges
            n = len(ranges)

            low_idx = int(round((low_angle - angle_min) / angle_inc))
            high_idx = int(round((high_angle - angle_min) / angle_inc))

            low_idx = max(0, min(n - 1, low_idx))
            high_idx = max(0, min(n - 1, high_idx))

            if low_idx > high_idx:
                low_idx, high_idx = high_idx, low_idx

            sector = [clean_range(r) for r in ranges[low_idx:high_idx + 1]]
            if len(sector) == 0:
                return self.laserscan.range_max

            return float(min(sector))

        self.take_measurements()

        if hasattr(self._pf, "visualize_particles"):
            self._pf.visualize_particles()
        if hasattr(self._pf, "visualize_estimate"):
            self._pf.visualize_estimate()

        close_count = 0
        stuck_turn_count = 0

        for step in range(max_steps):
            if rospy.is_shutdown():
                break

            if self.laserscan is None:
                self.rate.sleep()
                continue

            front_dist = get_min_range_between_angles(
                -math.radians(25.0),
                math.radians(25.0)
            )
            left_dist = get_min_range_between_angles(
                math.radians(25.0),
                math.radians(75.0)
            )
            right_dist = get_min_range_between_angles(
                -math.radians(75.0),
                -math.radians(25.0)
            )

            if front_dist < 0.35:
                close_count += 1
            else:
                close_count = 0

            rospy.loginfo(
                "LOCALIZE step=%d front=%.3f left=%.3f right=%.3f close_count=%d" %
                (step, front_dist, left_dist, right_dist, close_count)
            )

            if close_count >= 1:
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.6 if left_dist >= right_dist else -0.6

                start_time = rospy.Time.now().to_sec()
                while (rospy.Time.now().to_sec() - start_time) < 0.8 and (not rospy.is_shutdown()):
                    self.cmd_pub.publish(twist)
                    self.rate.sleep()

                self.cmd_pub.publish(Twist())
                stuck_turn_count += 1

            elif front_dist > 0.75:
                self.move_forward(0.18)
                stuck_turn_count = 0

            elif front_dist > 0.45:
                self.move_forward(0.10)
                stuck_turn_count = 0

            else:
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.45 if left_dist >= right_dist else -0.45

                start_time = rospy.Time.now().to_sec()
                while (rospy.Time.now().to_sec() - start_time) < 0.6 and (not rospy.is_shutdown()):
                    self.cmd_pub.publish(twist)
                    self.rate.sleep()

                self.cmd_pub.publish(Twist())
                stuck_turn_count += 1

            self.take_measurements()

            if hasattr(self._pf, "visualize_particles"):
                self._pf.visualize_particles()
            if hasattr(self._pf, "visualize_estimate"):
                self._pf.visualize_estimate()

            x_est, y_est, theta_est = self._pf.get_estimate()
            particles = getattr(self._pf, "_particles", None)

            if particles is not None and len(particles) > 0:
                xs = np.array([p.x for p in particles], dtype=np.float64)
                ys = np.array([p.y for p in particles], dtype=np.float64)
                thetas = np.array([p.theta for p in particles], dtype=np.float64)

                pos_spread = np.sqrt(np.var(xs) + np.var(ys))
                ang_spread = np.sqrt(np.var(np.sin(thetas)) + np.var(np.cos(thetas)))

                rospy.loginfo(
                    "PF estimate x=%.3f y=%.3f theta=%.3f pos_spread=%.3f ang_spread=%.3f" %
                    (x_est, y_est, theta_est, pos_spread, ang_spread)
                )

                if step > 15 and pos_spread < 0.18 and ang_spread < 0.35:
                    rospy.loginfo(
                        "PF converged near x=%.3f, y=%.3f, theta=%.3f" %
                        (x_est, y_est, theta_est)
                    )
                    break

            if stuck_turn_count > 8:
                rospy.logwarn("Too many turns during localization; stopping localization early.")
                break

            self.rate.sleep()

        self.cmd_pub.publish(Twist())
        rospy.loginfo("Localization phase complete.")

        ######### Your code ends here #########

        

    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########

        x_est, y_est, theta_est = self._pf.get_estimate()

        start_position = {
            "x": x_est,
            "y": y_est
        }

        rospy.loginfo(
            "Planning from PF estimate x=%.3f y=%.3f theta=%.3f to goal x=%.3f y=%.3f" %
            (
                x_est,
                y_est,
                theta_est,
                self.goal_position["x"],
                self.goal_position["y"],
            )
        )

        self.plan, graph = self._planner.generate_plan(start_position, self.goal_position)
        self.current_wp_idx = 0

        if self.plan is None or len(self.plan) == 0:
            rospy.logerr("RRT failed to find a path.")
            self.plan = []
            return

        if hasattr(self._planner, "visualize_graph"):
            self._planner.visualize_graph(graph)
        if hasattr(self._planner, "visualize_plan"):
            self._planner.visualize_plan(self.plan)

        rospy.loginfo("Generated plan with %d waypoints." % len(self.plan))

        for i, wp in enumerate(self.plan):
            rospy.loginfo("WP[%d] = x=%.3f y=%.3f" % (i, wp["x"], wp["y"]))

        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 3: Following the RRT path
    # ----------------------------------------------------------------------
    def follow_plan(self):
        """
        Follow the RRT waypoints using PID on (distance, heading) error.
        Keep updating PF along the way.
        """
        ######### Your code starts here #########

        def clean_range(r):
            if np.isnan(r) or np.isinf(r):
                return self.laserscan.range_max
            return r

        def get_min_range_between_angles(low_angle, high_angle):
            if self.laserscan is None:
                return self.laserscan.range_max

            angle_min = self.laserscan.angle_min
            angle_inc = self.laserscan.angle_increment
            ranges = self.laserscan.ranges
            n = len(ranges)

            low_idx = int(round((low_angle - angle_min) / angle_inc))
            high_idx = int(round((high_angle - angle_min) / angle_inc))

            low_idx = max(0, min(n - 1, low_idx))
            high_idx = max(0, min(n - 1, high_idx))

            if low_idx > high_idx:
                low_idx, high_idx = high_idx, low_idx

            sector = [clean_range(r) for r in ranges[low_idx:high_idx + 1]]
            if len(sector) == 0:
                return self.laserscan.range_max

            return float(min(sector))

        if self.plan is None or len(self.plan) == 0:
            rospy.logwarn("No plan available to follow.")
            return

        rospy.logwarn("=" * 50)
        rospy.logwarn("FOLLOWING PLAN WITH %d WAYPOINTS" % len(self.plan))
        for i, wp in enumerate(self.plan):
            rospy.logwarn("WP[%d]: x=%.3f y=%.3f" % (i, wp["x"], wp["y"]))
        rospy.logwarn("=" * 50)

        rate = rospy.Rate(20)

        current_wp_idx = 0
        iteration = 0

        MIN_DT = 1e-3
        last_pid_time = rospy.get_time()

        while not rospy.is_shutdown():
            if current_wp_idx >= len(self.plan):
                self.cmd_pub.publish(Twist())
                rospy.loginfo("Finished following plan.")
                break

            if self.current_position is None:
                rate.sleep()
                continue

            self.take_measurements()

            waypoint = self.plan[current_wp_idx]

            rx = self.current_position["x"]
            ry = self.current_position["y"]
            rtheta = self.current_position["theta"]

            dx = waypoint["x"] - rx
            dy = waypoint["y"] - ry

            distance_error = sqrt(dx * dx + dy * dy)

            if distance_error <= GOAL_THRESHOLD:
                rospy.loginfo("Reached waypoint %d / %d" % (current_wp_idx + 1, len(self.plan)))
                current_wp_idx += 1
                self.linear_pid.error_sum = 0.0
                self.angular_pid.error_sum = 0.0
                rate.sleep()
                continue

            desired_heading = atan2(dy, dx)
            heading_error = angle_to_neg_pi_to_pi(desired_heading - rtheta)

            front_dist = get_min_range_between_angles(
                -math.radians(25.0),
                math.radians(25.0)
            )
            left_dist = get_min_range_between_angles(
                math.radians(25.0),
                math.radians(80.0)
            )
            right_dist = get_min_range_between_angles(
                -math.radians(80.0),
                -math.radians(25.0)
            )

            now = rospy.get_time()
            if now <= last_pid_time:
                now = last_pid_time + MIN_DT
            last_pid_time = now

            linear_cmd = self.linear_pid.control(distance_error, now)
            angular_cmd = self.angular_pid.control(heading_error, now)

            if abs(heading_error) > 0.45:
                linear_cmd = 0.0

            linear_cmd = max(0.0, min(0.18, linear_cmd))
            angular_cmd = max(-1.2, min(1.2, angular_cmd))

            if front_dist < 0.28:
                linear_cmd = 0.0

                if left_dist >= right_dist:
                    angular_cmd = 0.65
                else:
                    angular_cmd = -0.65

                rospy.logwarn(
                    "SAFETY STOP: front=%.3f left=%.3f right=%.3f -> rotate ang=%.2f" %
                    (front_dist, left_dist, right_dist, angular_cmd)
                )

            elif front_dist < 0.40:
                linear_cmd = min(linear_cmd, 0.06)

            twist = Twist()
            twist.linear.x = linear_cmd
            twist.angular.z = angular_cmd
            self.cmd_pub.publish(twist)

            iteration += 1
            if iteration % 10 == 0:
                pf_x, pf_y, pf_t = self._pf.get_estimate()
                rospy.logwarn(
                    "iter=%d wp=%d odom=(%.2f,%.2f,%.2f) pf=(%.2f,%.2f,%.2f) "
                    "goal=(%.2f,%.2f) dist=%.2f heading_err=%.2f "
                    "front=%.2f left=%.2f right=%.2f cmd=(%.2f,%.2f)" %
                    (
                        iteration,
                        current_wp_idx,
                        rx,
                        ry,
                        rtheta,
                        pf_x,
                        pf_y,
                        pf_t,
                        waypoint["x"],
                        waypoint["y"],
                        distance_error,
                        heading_error,
                        front_dist,
                        left_dist,
                        right_dist,
                        linear_cmd,
                        angular_cmd,
                    )
                )

            rate.sleep()

        self.cmd_pub.publish(Twist())

        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Top-level
    # ----------------------------------------------------------------------
    def run(self):
        self.localize_with_pf()
        self.plan_with_rrt()
        self.follow_plan()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()

    with open(args.map_filepath, "r") as f:
        map_data = json.load(f)
        obstacles = map_data["obstacles"]
        map_aabb = map_data["map_aabb"]
        if "goal_position" not in map_data:
            raise RuntimeError("Map JSON must contain a 'goal_position' field.")
        goal_position = map_data["goal_position"]

    # Initialize ROS node
    rospy.init_node("pf_rrt_combined", anonymous=True)

    # Build map + PF + RRT
    map_obj = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.003
    rotation_variance = 0.03
    measurement_variance = 0.35

    pf = ParticleFilter(
        map_obj,
        num_particles,
        translation_variance,
        rotation_variance,
        measurement_variance,
    )
    planner = RrtPlanner(obstacles, map_aabb)

    controller = PFRRTController(pf, planner, goal_position)

    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
