#!/usr/bin/env python3
from typing import Optional, Dict, List, Tuple
from argparse import ArgumentParser
from math import sqrt, atan2, pi, inf
import math
import json
import random
import numpy as np

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

from lab8_9_starter import Map, ParticleFilter, angle_to_neg_pi_to_pi
from lab10_starter import RrtPlanner, PIDController as WaypointPID, GOAL_THRESHOLD


class PFRRTController:
    """
    Final project controller:
      1) actively localize with Particle Filter
      2) plan from PF estimate to known goal with RRT
      3) follow RRT waypoints while continuing PF updates

    Main changes for real maze:
      - avoid declaring PF convergence too early
      - use multi-beam validation, not only particle spread
      - use corner escape instead of repeated small turns
      - do not blindly drive forward when close to wall/corner
    """

    def __init__(self, pf: ParticleFilter, planner: RrtPlanner, goal_position: Dict[str, float]):
        self._pf = pf
        self._planner = planner
        self.goal_position = goal_position

        self.current_position: Optional[Dict[str, float]] = None
        self.last_odom: Optional[Dict[str, float]] = None
        self.laserscan: Optional[LaserScan] = None

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)

        # Conservative controller values for real robot maze.
        self.linear_pid = WaypointPID(0.30, 0.0, 0.08, 10, -0.16, 0.16)
        self.angular_pid = WaypointPID(0.65, 0.0, 0.12, 10, -1.10, 1.10)

        self.plan: Optional[List[Dict[str, float]]] = None
        self.current_wp_idx: int = 0
        self.rate = rospy.Rate(10)

        self.total_motion_distance = 0.0
        self.total_rotation_abs = 0.0

        while (self.current_position is None or self.laserscan is None) and (not rospy.is_shutdown()):
            rospy.loginfo("Waiting for /odom and /scan...")
            rospy.sleep(0.1)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        new_pose = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        if self.last_odom is not None:
            dx_world = new_pose["x"] - self.last_odom["x"]
            dy_world = new_pose["y"] - self.last_odom["y"]
            dtheta = angle_to_neg_pi_to_pi(new_pose["theta"] - self.last_odom["theta"])

            ct = math.cos(self.last_odom["theta"])
            st = math.sin(self.last_odom["theta"])
            dx_robot = ct * dx_world + st * dy_world
            dy_robot = -st * dx_world + ct * dy_world

            self._pf.move_by(dx_robot, dy_robot, dtheta)

        self.last_odom = new_pose
        self.current_position = new_pose

    def laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------
    def stop(self):
        self.cmd_pub.publish(Twist())
        rospy.sleep(0.05)

    def _valid_range(self, z: float) -> bool:
        return z is not None and (not np.isnan(z)) and (not np.isinf(z)) and z > 0.02

    def _range_at_angle(self, angle_rad: float) -> float:
        if self.laserscan is None:
            return inf
        angle_min = self.laserscan.angle_min
        angle_inc = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        n = len(ranges)
        if n == 0 or abs(angle_inc) < 1e-9:
            return inf
        idx = int(round((angle_rad - angle_min) / angle_inc))
        idx = max(0, min(n - 1, idx))
        z = ranges[idx]
        if self._valid_range(z):
            return float(z)
        return float(getattr(self.laserscan, "range_max", 3.5))

    def _sector_min(self, center_deg: float, width_deg: float) -> float:
        if self.laserscan is None:
            return inf
        angle_min = self.laserscan.angle_min
        angle_inc = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        n = len(ranges)
        if n == 0 or abs(angle_inc) < 1e-9:
            return inf

        lo = math.radians(center_deg - width_deg / 2.0)
        hi = math.radians(center_deg + width_deg / 2.0)
        lo_idx = int(round((lo - angle_min) / angle_inc))
        hi_idx = int(round((hi - angle_min) / angle_inc))
        lo_idx = max(0, min(n - 1, lo_idx))
        hi_idx = max(0, min(n - 1, hi_idx))
        if lo_idx > hi_idx:
            lo_idx, hi_idx = hi_idx, lo_idx
        vals = [float(r) for r in ranges[lo_idx:hi_idx + 1] if self._valid_range(r)]
        return min(vals) if vals else inf

    def _sector_mean(self, center_deg: float, width_deg: float) -> float:
        if self.laserscan is None:
            return inf
        angle_min = self.laserscan.angle_min
        angle_inc = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        n = len(ranges)
        if n == 0 or abs(angle_inc) < 1e-9:
            return inf

        lo = math.radians(center_deg - width_deg / 2.0)
        hi = math.radians(center_deg + width_deg / 2.0)
        lo_idx = int(round((lo - angle_min) / angle_inc))
        hi_idx = int(round((hi - angle_min) / angle_inc))
        lo_idx = max(0, min(n - 1, lo_idx))
        hi_idx = max(0, min(n - 1, hi_idx))
        if lo_idx > hi_idx:
            lo_idx, hi_idx = hi_idx, lo_idx
        vals = [float(r) for r in ranges[lo_idx:hi_idx + 1] if self._valid_range(r)]
        return float(np.mean(vals)) if vals else inf

    def _pf_spread(self) -> float:
        particles = getattr(self._pf, "_particles", [])
        if len(particles) == 0:
            return inf
        x_est, y_est, _ = self._pf.get_estimate()
        pts = np.array([[p.x, p.y] for p in particles])
        return float(np.std(np.linalg.norm(pts - np.array([x_est, y_est]), axis=1)))

    def _sensor_error_for_estimate(self) -> float:
        """
        Compare actual laser with predicted map distances from the PF estimate.
        Uses multiple beams. This is stricter than only checking front beam.
        """
        if self.laserscan is None:
            return inf
        x_est, y_est, theta_est = self._pf.get_estimate()
        beam_degs = [-90, -60, -30, 0, 30, 60, 90, 180]
        errors = []
        range_max = float(getattr(self.laserscan, "range_max", 3.5))

        for deg in beam_degs:
            a_robot = math.radians(deg)
            z = self._range_at_angle(a_robot)
            if not self._valid_range(z):
                continue
            z = min(z, range_max)
            pred = self._pf.map_.closest_distance((x_est, y_est), theta_est + a_robot)
            if pred is None:
                pred = range_max
            pred = min(float(pred), range_max)
            # Cap each beam's influence so one bad laser ray does not dominate.
            errors.append(min(abs(pred - z), 1.0))

        if not errors:
            return inf
        return float(np.mean(errors))

    def _visualize_pf(self):
        self._pf.visualize_particles()
        self._pf.visualize_estimate()

    def seed_particles_around(self, x: float, y: float, theta: float,
                              xy_std: float = 0.10, theta_std: float = 0.45):
        """
        Optional real-robot prior. Use this when the physical start position is
        approximately known. This prevents global PF from collapsing into a
        symmetric but wrong maze corner.
        """
        particles = getattr(self._pf, "_particles", [])
        if len(particles) == 0:
            return
        x_min, x_max, y_min, y_max = self._pf.map_.map_aabb
        logp = math.log(1.0 / float(len(particles)))
        def in_obstacle(px, py):
            for ox1, ox2, oy1, oy2 in self._pf.map_.obstacles:
                if ox1 <= px <= ox2 and oy1 <= py <= oy2:
                    return True
            return False
        for part in particles:
            for _ in range(80):
                px = random.gauss(x, xy_std)
                py = random.gauss(y, xy_std)
                if x_min <= px <= x_max and y_min <= py <= y_max and not in_obstacle(px, py):
                    part.x = px
                    part.y = py
                    break
            else:
                part.x = min(max(x, x_min), x_max)
                part.y = min(max(y, y_min), y_max)
            part.theta = angle_to_neg_pi_to_pi(random.gauss(theta, theta_std))
            part.log_p = logp
        if hasattr(self._pf, "update_count"):
            self._pf.update_count = 0
        rospy.loginfo(
            f"Seeded PF around start hint: x={x:.2f}, y={y:.2f}, theta={theta:.2f}, "
            f"xy_std={xy_std:.2f}, theta_std={theta_std:.2f}"
        )
        self._visualize_pf()

    # ------------------------------------------------------------------
    # Low-level motion primitives with front safety
    # ------------------------------------------------------------------
    def move_forward(self, distance: float, stop_distance: float = 0.24):
        speed = 0.12 if distance >= 0 else -0.10
        duration = abs(distance) / max(abs(speed), 1e-6)
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(20)

        cmd = Twist()
        cmd.linear.x = speed

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            if distance > 0:
                front = self._sector_min(0.0, 35.0)
                if front < stop_distance:
                    rospy.loginfo(f"move_forward safety stop: front={front:.2f}")
                    break
            self.cmd_pub.publish(cmd)
            r.sleep()

        self.stop()
        self.total_motion_distance += abs(distance)

    def rotate_in_place(self, angle: float):
        angular_speed = 0.65
        duration = abs(angle) / angular_speed if angular_speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(20)

        cmd = Twist()
        cmd.angular.z = angular_speed if angle >= 0.0 else -angular_speed

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(cmd)
            r.sleep()

        self.stop()
        self.total_rotation_abs += abs(angle)

    # ------------------------------------------------------------------
    # PF measurement update
    # ------------------------------------------------------------------
    def take_measurements(self, full: bool = True):
        """
        Update PF using laser beams.

        Important: ParticleFilter.measure() may resample internally. Therefore we
        do not use too many beams every cycle. For localization we use enough
        distinct directions to break symmetry; for following we use fewer beams.
        """
        if self.laserscan is None:
            return

        range_max = float(getattr(self.laserscan, "range_max", 3.5))
        if full:
            beam_degs = [-90, -45, -20, 0, 20, 45, 90, 180]
        else:
            beam_degs = [-45, 0, 45]

        random.shuffle(beam_degs)
        for deg in beam_degs:
            a = math.radians(deg)
            z = self._range_at_angle(a)
            if not self._valid_range(z):
                z = range_max
            z = min(float(z), range_max)
            self._pf.measure(z, a)

    # ------------------------------------------------------------------
    # Active exploration helpers
    # ------------------------------------------------------------------
    def escape_corner(self):
        """
        Real robot can get trapped in the upper-right corner and just wiggle.
        This routine makes a decisive backup + large turn + short exit move.
        """
        front = self._sector_min(0.0, 45.0)
        left = self._sector_mean(75.0, 50.0)
        right = self._sector_mean(-75.0, 50.0)
        rospy.loginfo(f"Corner escape: front={front:.2f}, left={left:.2f}, right={right:.2f}")

        self.stop()
        self.move_forward(-0.16)
        self.take_measurements(full=True)

        # Turn toward more open side. Use a large angle; small turns are what
        # cause the robot to wiggle in the corner.
        if left >= right:
            self.rotate_in_place(math.radians(85.0))
        else:
            self.rotate_in_place(-math.radians(85.0))
        self.take_measurements(full=True)

        if self._sector_min(0.0, 35.0) > 0.42:
            self.move_forward(0.20, stop_distance=0.27)
            self.take_measurements(full=True)

    def _active_localization_step(self) -> bool:
        """
        Return True if this step performed a corner escape.
        """
        front = self._sector_min(0.0, 40.0)
        left = self._sector_mean(65.0, 55.0)
        right = self._sector_mean(-65.0, 55.0)
        left_close = self._sector_min(60.0, 55.0)
        right_close = self._sector_min(-60.0, 55.0)

        # Strong corner condition: front is close and at least one side is also close.
        if front < 0.34 and min(left_close, right_close) < 0.32:
            self.escape_corner()
            return True

        # If front is clear, move a small amount. Short moves avoid wall contact.
        if front > 0.72:
            self.move_forward(0.18, stop_distance=0.30)
            return False

        # Front partly blocked: turn toward the more open side.
        if left > right:
            self.rotate_in_place(math.radians(random.uniform(45.0, 75.0)))
        else:
            self.rotate_in_place(-math.radians(random.uniform(45.0, 75.0)))
        return False

    # ------------------------------------------------------------------
    # Phase 1: Localization with PF
    # ------------------------------------------------------------------
    def localize_with_pf(self, max_steps: int = 260):
        rospy.loginfo("Starting PF localization with anti-false-convergence exploration...")

        # First collect a 360-degree signature. This helps avoid immediately
        # collapsing particles based on only one wall direction.
        for _ in range(8):
            if rospy.is_shutdown():
                return
            self.take_measurements(full=True)
            self.rotate_in_place(math.radians(45.0))
            self._visualize_pf()

        good_count = 0
        escape_count = 0
        no_progress_turns = 0
        last_est = None
        rate = rospy.Rate(2)

        for step in range(max_steps):
            if rospy.is_shutdown():
                break

            did_escape = self._active_localization_step()
            if did_escape:
                escape_count += 1
                no_progress_turns = 0
            else:
                front = self._sector_min(0.0, 40.0)
                if front < 0.50:
                    no_progress_turns += 1
                else:
                    no_progress_turns = 0

            if no_progress_turns >= 4:
                rospy.loginfo("Localization: repeated local turns; forcing corner escape.")
                self.escape_corner()
                escape_count += 1
                no_progress_turns = 0

            self.take_measurements(full=True)
            self._visualize_pf()

            x_est, y_est, theta_est = self._pf.get_estimate()
            spread = self._pf_spread()
            sensor_error = self._sensor_error_for_estimate()
            front_now = self._sector_min(0.0, 35.0)

            if last_est is None:
                est_jump = 0.0
            else:
                est_jump = sqrt((x_est - last_est[0]) ** 2 + (y_est - last_est[1]) ** 2)
            last_est = (x_est, y_est)

            # Conservative convergence. The previous failure was spread becoming
            # tiny while the robot was still in the wrong symmetric corner.
            # Requirements:
            #   - not too early
            #   - enough actual exploration motion
            #   - particle cloud compact
            #   - multi-beam map/laser error small for several consecutive cycles
            #   - estimate not jumping around
            can_consider_converged = (
                step >= 35 and
                self.total_motion_distance >= 0.65 and
                self.total_rotation_abs >= 2.5 * math.pi
            )
            if can_consider_converged and spread < 0.10 and sensor_error < 0.20 and est_jump < 0.08:
                good_count += 1
            else:
                good_count = 0

            rospy.loginfo(
                f"PF step {step}: spread={spread:.3f}, sensor_error={sensor_error:.3f}, "
                f"good={good_count}, front={front_now:.2f}, moved={self.total_motion_distance:.2f}, "
                f"escapes={escape_count}"
            )

            if good_count >= 6:
                rospy.loginfo("PF localization converged.")
                break

            rate.sleep()

        self.stop()

    # ------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ------------------------------------------------------------------
    def _dist(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

    def _line_is_free(self, a: Dict[str, float], b: Dict[str, float], step: float = 0.04) -> bool:
        # Use planner's collision check if available.
        try:
            from lab10_starter import Node
        except Exception:
            Node = None

        d = self._dist(a, b)
        checks = max(2, int(math.ceil(d / step)))
        for i in range(checks + 1):
            t = i / float(checks)
            x = a["x"] + t * (b["x"] - a["x"])
            y = a["y"] + t * (b["y"] - a["y"])
            if Node is not None and hasattr(self._planner, "_is_in_collision"):
                if self._planner._is_in_collision(Node(np.array([x, y]), None)):
                    return False
            else:
                for obs in self._planner.obstacles:
                    ox1, ox2, oy1, oy2 = obs
                    pad = getattr(self._planner, "obstacle_padding", 0.15)
                    if ox1 - pad <= x <= ox2 + pad and oy1 - pad <= y <= oy2 + pad:
                        return False
        return True

    def _shortcut_plan(self, plan: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if plan is None or len(plan) <= 2:
            return plan
        result = [plan[0]]
        i = 0
        while i < len(plan) - 1:
            j = len(plan) - 1
            chosen = i + 1
            while j > i + 1:
                if self._line_is_free(plan[i], plan[j]):
                    chosen = j
                    break
                j -= 1
            result.append(plan[chosen])
            i = chosen
        return result

    def plan_with_rrt(self):
        rospy.loginfo("Planning path with RRT...")
        x_est, y_est, theta_est = self._pf.get_estimate()
        start = {"x": float(x_est), "y": float(y_est), "theta": float(theta_est)}
        goal = {"x": float(self.goal_position["x"]), "y": float(self.goal_position["y"])}

        rospy.loginfo(
            f"RRT start from PF estimate ({start['x']:.3f}, {start['y']:.3f}, {start['theta']:.2f}) "
            f"to goal ({goal['x']:.3f}, {goal['y']:.3f})"
        )

        plan, graph = self._planner.generate_plan(start, goal)
        if plan is None or len(plan) == 0:
            raise RuntimeError("RRT failed to generate a plan.")

        # Remove duplicate points and shortcut where safe.
        cleaned = []
        for wp in plan:
            wp2 = {"x": float(wp["x"]), "y": float(wp["y"])}
            if not cleaned or self._dist(cleaned[-1], wp2) > 0.05:
                cleaned.append(wp2)
        self.plan = self._shortcut_plan(cleaned)
        self.current_wp_idx = 0

        self._planner.visualize_plan(self.plan)
        self._planner.visualize_graph(graph)
        rospy.loginfo(f"RRT generated {len(plan)} waypoints; following {len(self.plan)} after shortcutting.")

    # ------------------------------------------------------------------
    # Phase 3: Follow plan
    # ------------------------------------------------------------------
    def follow_plan(self):
        if self.plan is None or len(self.plan) == 0:
            raise RuntimeError("No plan available. Call plan_with_rrt() first.")

        rospy.loginfo("Following RRT plan...")
        self.current_wp_idx = 0
        last_pf_update = rospy.Time.now().to_sec()
        stuck_count = 0
        prev_dist = None

        while not rospy.is_shutdown() and self.current_wp_idx < len(self.plan):
            now = rospy.Time.now().to_sec()
            if now - last_pf_update > 0.35:
                self.take_measurements(full=False)
                self._visualize_pf()
                last_pf_update = now

            x, y, theta = self._pf.get_estimate()
            wp = self.plan[self.current_wp_idx]
            dx = wp["x"] - x
            dy = wp["y"] - y
            dist = sqrt(dx * dx + dy * dy)
            desired_theta = atan2(dy, dx)
            angle_error = angle_to_neg_pi_to_pi(desired_theta - theta)

            threshold = GOAL_THRESHOLD if self.current_wp_idx == len(self.plan) - 1 else max(GOAL_THRESHOLD, 0.14)
            if dist < threshold:
                rospy.loginfo(f"Reached waypoint {self.current_wp_idx + 1}/{len(self.plan)}")
                self.current_wp_idx += 1
                self.stop()
                prev_dist = None
                continue

            # Detect lack of progress near a waypoint.
            if prev_dist is not None and dist > prev_dist - 0.01:
                stuck_count += 1
            else:
                stuck_count = 0
            prev_dist = dist

            if stuck_count >= 12:
                rospy.loginfo("Follow-plan stuck near waypoint; performing small escape and replanning.")
                self.escape_corner()
                self.plan_with_rrt()
                stuck_count = 0
                prev_dist = None
                continue

            front = self._sector_min(0.0, 35.0)
            left = self._sector_mean(65.0, 50.0)
            right = self._sector_mean(-65.0, 50.0)

            # Emergency obstacle handling.
            if front < 0.20:
                rospy.loginfo(f"Path following safety: front={front:.2f}; backing up and replanning.")
                self.stop()
                self.move_forward(-0.12)
                if left >= right:
                    self.rotate_in_place(math.radians(55.0))
                else:
                    self.rotate_in_place(-math.radians(55.0))
                self.take_measurements(full=True)
                self.plan_with_rrt()
                continue

            t = rospy.get_time()
            linear_cmd = self.linear_pid.control(dist, t)
            angular_cmd = self.angular_pid.control(angle_error, t)

            # Rotate-first behavior for narrow maze.
            if abs(angle_error) > 0.55:
                linear_cmd = 0.0
            elif abs(angle_error) > 0.28:
                linear_cmd = min(linear_cmd, 0.055)

            # Slow down near walls even if path says go forward.
            if front < 0.35:
                linear_cmd = min(linear_cmd, 0.045)
            elif front < 0.50:
                linear_cmd = min(linear_cmd, 0.075)

            cmd = Twist()
            cmd.linear.x = float(max(0.0, min(0.14, linear_cmd)))
            cmd.angular.z = float(max(-1.05, min(1.05, angular_cmd)))
            self.cmd_pub.publish(cmd)

            rospy.loginfo(
                f"wp {self.current_wp_idx + 1}/{len(self.plan)} | dist={dist:.3f}, "
                f"angle={angle_error:.3f}, front={front:.2f}, "
                f"v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}, spread={self._pf_spread():.3f}"
            )
            self.rate.sleep()

        self.stop()
        rospy.loginfo("Finished following plan.")

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------
    def run(self):
        self.localize_with_pf()
        self.plan_with_rrt()
        self.follow_plan()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    parser.add_argument("--start_x", type=float, default=None)
    parser.add_argument("--start_y", type=float, default=None)
    parser.add_argument("--start_theta", type=float, default=0.0)
    parser.add_argument("--start_std", type=float, default=0.10)
    args = parser.parse_args()

    with open(args.map_filepath, "r") as f:
        map_data = json.load(f)
        obstacles = map_data["obstacles"]
        map_aabb = map_data["map_aabb"]
        if "goal_position" not in map_data:
            raise RuntimeError("Map JSON must contain a 'goal_position' field.")
        goal_position = map_data["goal_position"]

    rospy.init_node("pf_rrt_combined", anonymous=True)

    map_obj = Map(obstacles, map_aabb)

    # These values intentionally keep PF less over-confident on real robot.
    # If particles still collapse to the wrong corner, increase measurement_variance
    # to 0.65 or increase num_particles to 500.
    num_particles = 400
    translation_variance = 0.006
    rotation_variance = 0.06
    measurement_variance = 0.55

    pf = ParticleFilter(
        map_obj,
        num_particles,
        translation_variance,
        rotation_variance,
        measurement_variance,
    )
    planner = RrtPlanner(obstacles, map_aabb)

    # Slightly conservative RRT padding for real walls.
    if hasattr(planner, "obstacle_padding"):
        planner.obstacle_padding = 0.17
    if hasattr(planner, "delta"):
        planner.delta = 0.10

    controller = PFRRTController(pf, planner, goal_position)

    if args.start_x is not None and args.start_y is not None:
        controller.seed_particles_around(
            args.start_x, args.start_y, args.start_theta,
            xy_std=args.start_std, theta_std=0.45
        )

    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
