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

from lab8_9_starter import Map, ParticleFilter, angle_to_neg_pi_to_pi
from lab10_starter import RrtPlanner, GOAL_THRESHOLD


class SafePID:
    def __init__(self, kP, kI, kD, err_int_max, u_min, u_max):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.err_int_max = abs(err_int_max)
        self.u_min = u_min
        self.u_max = u_max
        self.err_prev = None
        self.t_prev = None
        self.err_int = 0.0

    def reset(self):
        self.err_prev = None
        self.t_prev = None
        self.err_int = 0.0

    def control(self, err, t):
        if self.t_prev is None:
            dt = 0.05
            derr = 0.0
        else:
            dt = max(float(t - self.t_prev), 1e-3)
            derr = (err - self.err_prev) / dt

        self.err_int += err * dt
        self.err_int = max(-self.err_int_max, min(self.err_int_max, self.err_int))

        u = self.kP * err + self.kI * self.err_int + self.kD * derr
        u = max(self.u_min, min(self.u_max, u))

        self.err_prev = err
        self.t_prev = t
        return u


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
        self.linear_pid = SafePID(0.30, 0.0, 0.04, 10, -0.20, 0.20)
        self.angular_pid = SafePID(0.85, 0.0, 0.08, 10, -1.20, 1.20)

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


    def get_sector_distance(self, center_deg: float = 0.0, half_width_deg: float = 10.0, percentile: float = 25.0) -> float:
        if self.laserscan is None:
            return inf

        angle_min = self.laserscan.angle_min
        angle_inc = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        n = len(ranges)
        if n == 0 or angle_inc == 0:
            return inf

        low_angle = math.radians(center_deg - half_width_deg)
        high_angle = math.radians(center_deg + half_width_deg)
        low_idx = int(round((low_angle - angle_min) / angle_inc))
        high_idx = int(round((high_angle - angle_min) / angle_inc))
        low_idx = max(0, min(n - 1, low_idx))
        high_idx = max(0, min(n - 1, high_idx))
        if low_idx > high_idx:
            low_idx, high_idx = high_idx, low_idx

        vals = []
        rmin = getattr(self.laserscan, "range_min", 0.02)
        rmax = getattr(self.laserscan, "range_max", 10.0)
        for r in ranges[low_idx:high_idx + 1]:
            if np.isfinite(r) and rmin <= r <= rmax:
                vals.append(float(r))
        if not vals:
            return inf
        return float(np.percentile(vals, percentile))

    def get_front_distance(self) -> float:
        return self.get_sector_distance(0.0, 9.0, 25.0)

    def get_wide_front_distance(self) -> float:
        return self.get_sector_distance(0.0, 18.0, 20.0)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

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
    def localize_with_pf(self, max_steps: int = 260):
        """
        Localize with short, cautious motions. The robot does not do a blind 360-degree
        spin or a long initial forward drive.
        """
        ######### Your code starts here #########
        rospy.loginfo("Starting particle-filter localization...")

        for _ in range(8):
            if rospy.is_shutdown():
                return
            self.take_measurements()
            self._pf.visualize_particles()
            self._pf.visualize_estimate()
            rospy.sleep(0.06)

        stuck_count = 0
        last_estimate = None
        rate = rospy.Rate(3)

        for step in range(max_steps):
            if rospy.is_shutdown():
                break

            front = self.get_front_distance()
            wide_front = self.get_wide_front_distance()

            if front < 0.14:
                rospy.loginfo("Obstacle very close during localization; backing up a little.")
                self.move_forward(-0.08)
                self.rotate_in_place(np.random.uniform(0.35, 0.65))
            elif wide_front < 0.24:
                rospy.loginfo("Obstacle ahead during localization; turning cautiously.")
                self.rotate_in_place(np.random.uniform(0.30, 0.55))
            elif front > 0.55:
                self.move_forward(0.10)
            else:
                self.rotate_in_place(np.random.uniform(-0.35, 0.35))

            for _ in range(2):
                self.take_measurements()

            self._pf.visualize_particles()
            self._pf.visualize_estimate()

            x_est, y_est, theta_est = self._pf.get_estimate()
            particles = getattr(self._pf, "_particles", [])
            if len(particles) > 0:
                pts = np.array([[p.x, p.y] for p in particles])
                spread = float(np.std(np.linalg.norm(pts - np.array([x_est, y_est]), axis=1)))

                sensor_ok = True
                if np.isfinite(front):
                    predicted = self._pf.map_.closest_distance((x_est, y_est), theta_est)
                    if predicted is not None:
                        sensor_ok = abs(predicted - front) < 0.45

                if last_estimate is not None:
                    moved = sqrt((x_est - last_estimate[0]) ** 2 + (y_est - last_estimate[1]) ** 2)
                    if moved < 0.015 and spread > 0.22:
                        stuck_count += 1
                    else:
                        stuck_count = 0
                last_estimate = (x_est, y_est, theta_est)

                if stuck_count >= 4:
                    self.rotate_in_place(np.random.uniform(0.45, 0.75))
                    stuck_count = 0

                rospy.loginfo(f"PF localization step {step}: spread={spread:.3f}, sensor_ok={sensor_ok}, front={front:.3f}")
                if spread < 0.17 and sensor_ok and step > 6:
                    rospy.loginfo("Particle filter localization converged.")
                    break

            rate.sleep()

        self.stop_robot()
        ######### Your code ends here #########


    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########
        rospy.loginfo("Planning path with RRT...")

        x_est, y_est, theta_est = self._pf.get_estimate()
        start = {"x": float(x_est), "y": float(y_est), "theta": float(theta_est)}
        goal = {"x": float(self.goal_position["x"]), "y": float(self.goal_position["y"])}

        plan, graph = self._planner.generate_plan(start, goal)

        # Light path simplification: remove nearly duplicate points and almost
        # collinear waypoints to reduce unnecessary small turns.
        simplified = []
        for wp in plan:
            if not simplified:
                simplified.append(wp)
                continue
            prev = simplified[-1]
            if sqrt((wp["x"] - prev["x"]) ** 2 + (wp["y"] - prev["y"]) ** 2) >= 0.05:
                simplified.append(wp)

        if len(simplified) >= 3:
            pruned = [simplified[0]]
            for i in range(1, len(simplified) - 1):
                a = pruned[-1]
                b = simplified[i]
                c = simplified[i + 1]
                v1 = np.array([b["x"] - a["x"], b["y"] - a["y"]])
                v2 = np.array([c["x"] - b["x"], c["y"] - b["y"]])
                if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                    continue
                cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                cosang = max(-1.0, min(1.0, cosang))
                angle_change = abs(math.acos(cosang))
                if angle_change > 0.20:
                    pruned.append(b)
            pruned.append(simplified[-1])
            simplified = pruned

        self.plan = simplified
        self.current_wp_idx = 0

        self._planner.visualize_plan(self.plan)
        self._planner.visualize_graph(graph)

        rospy.loginfo(f"RRT generated {len(plan)} waypoints; using {len(self.plan)} after simplification.")
        if len(self.plan) == 0:
            raise RuntimeError("RRT failed to generate a non-empty plan.")
        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 3: Following the RRT path
    # ----------------------------------------------------------------------
    def follow_plan(self):
        """
        Follow RRT waypoints using a smoothed PF estimate and robust front safety.
        """
        ######### Your code starts here #########
        if self.plan is None or len(self.plan) == 0:
            raise RuntimeError("No plan available. Call plan_with_rrt() before follow_plan().")

        rospy.loginfo("Following RRT plan...")
        self.current_wp_idx = 0
        self.linear_pid.reset()
        self.angular_pid.reset()
        last_measurement_time = rospy.Time.now().to_sec()
        filtered_pose = None
        emergency_count = 0

        while not rospy.is_shutdown() and self.current_wp_idx < len(self.plan):
            now = rospy.Time.now().to_sec()
            if now - last_measurement_time > 0.35:
                self.take_measurements()
                self._pf.visualize_particles()
                self._pf.visualize_estimate()
                last_measurement_time = now

            x_raw, y_raw, theta_raw = self._pf.get_estimate()
            if filtered_pose is None:
                filtered_pose = [x_raw, y_raw, theta_raw]
            else:
                alpha = 0.35
                filtered_pose[0] = (1.0 - alpha) * filtered_pose[0] + alpha * x_raw
                filtered_pose[1] = (1.0 - alpha) * filtered_pose[1] + alpha * y_raw
                dtheta = angle_to_neg_pi_to_pi(theta_raw - filtered_pose[2])
                filtered_pose[2] = angle_to_neg_pi_to_pi(filtered_pose[2] + alpha * dtheta)

            x, y, theta = filtered_pose
            waypoint = self.plan[self.current_wp_idx]

            dx = waypoint["x"] - x
            dy = waypoint["y"] - y
            distance_error = sqrt(dx ** 2 + dy ** 2)
            desired_theta = atan2(dy, dx)
            angle_error = angle_to_neg_pi_to_pi(desired_theta - theta)

            threshold = GOAL_THRESHOLD if self.current_wp_idx == len(self.plan) - 1 else max(GOAL_THRESHOLD, 0.16)
            if distance_error < threshold:
                rospy.loginfo(f"Reached waypoint {self.current_wp_idx + 1}/{len(self.plan)}")
                self.current_wp_idx += 1
                self.linear_pid.reset()
                self.angular_pid.reset()
                self.stop_robot()
                continue

            t_now = rospy.get_time()
            linear_cmd = self.linear_pid.control(distance_error, t_now)
            angular_cmd = self.angular_pid.control(angle_error, t_now)

            if abs(angle_error) > 0.70:
                linear_cmd = 0.0
            elif abs(angle_error) > 0.35:
                linear_cmd = min(linear_cmd, 0.06)

            front = self.get_front_distance()
            wide_front = self.get_wide_front_distance()

            if front < 0.12 and linear_cmd > 0.0:
                emergency_count += 1
            else:
                emergency_count = 0

            if emergency_count >= 2:
                rospy.loginfo("Obstacle very close while following path; backing up and replanning.")
                self.stop_robot()
                self.move_forward(-0.08)
                self.plan_with_rrt()
                self.linear_pid.reset()
                self.angular_pid.reset()
                filtered_pose = None
                emergency_count = 0
                continue

            if wide_front < 0.22 and linear_cmd > 0.0:
                linear_cmd = min(linear_cmd, 0.035)

            cmd = Twist()
            cmd.linear.x = float(max(0.0, min(0.16, linear_cmd)))
            cmd.angular.z = float(max(-1.0, min(1.0, angular_cmd)))
            self.cmd_pub.publish(cmd)

            rospy.loginfo(
                f"wp {self.current_wp_idx + 1}/{len(self.plan)} | "
                f"dist={distance_error:.3f}, angle={angle_error:.3f}, "
                f"front={front:.3f}, v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}"
            )
            self.rate.sleep()

        self.stop_robot()
        rospy.loginfo("Finished following plan.")
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
    num_particles = 400
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
