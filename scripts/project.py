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
from lab8_9 import Map, ParticleFilter, angle_to_neg_pi_to_pi  # :contentReference[oaicite:2]{index=2}
from lab10 import RrtPlanner, PIDController as WaypointPID, GOAL_THRESHOLD  # :contentReference[oaicite:3]{index=3}


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
        self.take_measurements()

        if hasattr(self._pf, "visualize_particles"):
            self._pf.visualize_particles()
        if hasattr(self._pf, "visualize_estimate"):
            self._pf.visualize_estimate()
    
        for step in range(max_steps):
            if rospy.is_shutdown():
                break
    
            est_x, est_y, est_theta = self._pf.get_estimate()
    
            converged = False
            particles = getattr(self._pf, "_particles", None)
            if particles is not None and len(particles) > 0:
                xs = np.array([p.x for p in particles], dtype=np.float64)
                ys = np.array([p.y for p in particles], dtype=np.float64)
                thetas = np.array([p.theta for p in particles], dtype=np.float64)
    
                pos_spread = np.sqrt(np.var(xs) + np.var(ys))
                ang_spread = np.sqrt(np.var(np.sin(thetas)) + np.var(np.cos(thetas)))
    
                if pos_spread < 0.08 and ang_spread < 0.30 and step > 10:
                    converged = True
    
            if converged:
                rospy.loginfo(
                    f"PF converged near x={est_x:.3f}, y={est_y:.3f}, theta={est_theta:.3f}"
                )
                break
    
            if self.laserscan is None:
                self.rate.sleep()
                continue
    
            angle_increment = self.laserscan.angle_increment
            ranges = np.array(self.laserscan.ranges, dtype=np.float64)
            ranges[np.isinf(ranges)] = self.laserscan.range_max
            ranges[np.isnan(ranges)] = self.laserscan.range_max
    
            num_ranges = len(ranges)
            mid_idx = num_ranges // 2
    
            ten_deg = max(1, int((10.0 * math.pi / 180.0) / angle_increment))
            forty_five_deg = max(1, int((45.0 * math.pi / 180.0) / angle_increment))
    
            front_low = max(0, mid_idx - ten_deg)
            front_high = min(num_ranges, mid_idx + ten_deg + 1)
    
            left_low = min(num_ranges - 1, mid_idx + ten_deg)
            left_high = min(num_ranges, mid_idx + forty_five_deg + 1)
    
            right_low = max(0, mid_idx - forty_five_deg)
            right_high = max(1, mid_idx - ten_deg + 1)
    
            front_dist = float(np.min(ranges[front_low:front_high])) if front_high > front_low else self.laserscan.range_max
            left_dist = float(np.min(ranges[left_low:left_high])) if left_high > left_low else self.laserscan.range_max
            right_dist = float(np.min(ranges[right_low:right_high])) if right_high > right_low else self.laserscan.range_max
    
            if front_dist > 0.60:
                self.move_forward(0.20)
            elif front_dist > 0.35:
                self.move_forward(0.10)
            else:
                self.move_forward(-0.08)
                if left_dist >= right_dist:
                    self.rotate_in_place(pi / 3.0)
                else:
                    self.rotate_in_place(-pi / 3.0)
    
            rospy.sleep(0.1)
            self.take_measurements()
    
            if hasattr(self._pf, "visualize_particles"):
                self._pf.visualize_particles()
            if hasattr(self._pf, "visualize_estimate"):
                self._pf.visualize_estimate()
        ######### Your code ends here #########

        

    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########
        est_x, est_y, est_theta = self._pf.get_estimate()
        start_position = {"x": est_x, "y": est_y}
    
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
    
        rospy.loginfo(f"Generated plan with {len(self.plan)} waypoints")
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
        if self.plan is None or len(self.plan) == 0:
            rospy.logwarn("No plan available to follow.")
            return
    
        while not rospy.is_shutdown() and self.current_wp_idx < len(self.plan):
            self.take_measurements()
            est_x, est_y, est_theta = self._pf.get_estimate()
    
            waypoint = self.plan[self.current_wp_idx]
            dx = waypoint["x"] - est_x
            dy = waypoint["y"] - est_y
            distance_error = sqrt(dx * dx + dy * dy)
    
            if distance_error <= GOAL_THRESHOLD:
                self.current_wp_idx += 1
                continue
    
            desired_heading = atan2(dy, dx)
            heading_error = angle_to_neg_pi_to_pi(desired_heading - est_theta)
    
            now = rospy.Time.now().to_sec()
            angular_cmd = self.angular_pid.control(heading_error, now)
    
            if abs(heading_error) > 0.35:
                linear_cmd = 0.0
            else:
                linear_cmd = self.linear_pid.control(distance_error, now)
                linear_cmd = max(0.0, min(0.20, linear_cmd))
    
            if self.laserscan is not None:
                ranges = np.array(self.laserscan.ranges, dtype=np.float64)
                ranges[np.isinf(ranges)] = self.laserscan.range_max
                ranges[np.isnan(ranges)] = self.laserscan.range_max
    
                mid_idx = len(ranges) // 2
                angle_increment = self.laserscan.angle_increment
                ten_deg = max(1, int((10.0 * math.pi / 180.0) / angle_increment))
                front_low = max(0, mid_idx - ten_deg)
                front_high = min(len(ranges), mid_idx + ten_deg + 1)
                front_dist = float(np.min(ranges[front_low:front_high]))
    
                if front_dist < 0.18:
                    linear_cmd = 0.0
                    angular_cmd = 0.6 if heading_error >= 0.0 else -0.6
    
            twist = Twist()
            twist.linear.x = linear_cmd
            twist.angular.z = angular_cmd
            self.cmd_pub.publish(twist)
            self.rate.sleep()
    
        self.cmd_pub.publish(Twist())
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
