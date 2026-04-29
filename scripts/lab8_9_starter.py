#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import inf, sqrt, atan2, pi
from time import sleep, time
import queue
import json
import math
from random import uniform
import copy

import scipy
import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Twist, Point32, PoseStamped, Pose, Vector3, Quaternion, Point, PoseArray
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import scipy.stats
from numpy.random import choice

np.set_printoptions(linewidth=200)

# AABB format: (x_min, x_max, y_min, y_max)
OBS_TYPE = Tuple[float, float, float, float]
# Position format: {"x": x, "y": y, "theta": theta}
POSITION_TYPE = Dict[str, float]

# don't change this
GOAL_THRESHOLD = 0.1


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


def angle_to_neg_pi_to_pi(angle: float) -> float:
    while angle < -pi:
        angle += 2 * pi
    while angle > pi:
        angle -= 2 * pi
    return angle


# see https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
def ray_line_intersection(ray_origin, ray_direction_rad, point1, point2):
    # Convert to numpy arrays
    ray_origin = np.array(ray_origin, dtype=np.float32)
    ray_direction = np.array([math.cos(ray_direction_rad), math.sin(ray_direction_rad)])
    point1 = np.array(point1, dtype=np.float32)
    point2 = np.array(point2, dtype=np.float32)

    # Ray-Line Segment Intersection Test in 2D
    v1 = ray_origin - point1
    v2 = point2 - point1
    v3 = np.array([-ray_direction[1], ray_direction[0]])
    denominator = np.dot(v2, v3)
    if denominator == 0:
        return None
    t1 = np.cross(v2, v1) / denominator
    t2 = np.dot(v1, v3) / denominator
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return [ray_origin + t1 * ray_direction]
    return None


class Map:
    def __init__(self, obstacles: List[OBS_TYPE], map_aabb: Tuple):
        self.obstacles = obstacles
        self.map_aabb = map_aabb

    @property
    def top_right(self) -> Tuple[float, float]:
        return self.map_aabb[1], self.map_aabb[3]

    @property
    def bottom_left(self) -> Tuple[float, float]:
        return self.map_aabb[0], self.map_aabb[2]

    def draw_distances(self, origins: List[Tuple[float, float]]):
        """Example usage:
        map_ = Map(obstacles, map_aabb)
        map_.draw_distances([(0.0, 0.0), (3, 3), (1.5, 1.5)])
        """

        # Draw scene
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.tight_layout()
        x_min_global, x_max_global, y_min_global, y_max_global = self.map_aabb
        for aabb in self.obstacles:
            width = aabb[1] - aabb[0]
            height = aabb[3] - aabb[2]
            rect = patches.Rectangle(
                (aabb[0], aabb[2]), width, height, linewidth=2, edgecolor="r", facecolor="r", alpha=0.4
            )
            ax.add_patch(rect)
        ax.set_xlim(x_min_global, x_max_global)
        ax.set_ylim(y_min_global, y_max_global)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D Plot of Obstacles")
        ax.set_aspect("equal", "box")
        plt.grid(True)

        # Draw rays
        angles = np.linspace(0, 2 * math.pi, 10, endpoint=False)
        for origin in origins:
            for angle in angles:
                closest_distance = self.closest_distance(origin, angle)
                if closest_distance is not None:
                    x = origin[0] + closest_distance * math.cos(angle)
                    y = origin[1] + closest_distance * math.sin(angle)
                    plt.plot([origin[0], x], [origin[1], y], "b-")
        plt.show()

    def closest_distance(self, origin: Tuple[float, float], angle: float) -> Optional[float]:
        """Returns the closest distance to an obstacle from the given origin in the given direction `angle`. If no
        intersection is found, returns `None`.
        """

        def lines_from_obstacle(obstacle: OBS_TYPE):
            """Returns the four lines of the given AABB format obstacle.
            Example usage: `point0, point1 = lines_from_obstacle(self.obstacles[0])`
            """
            x_min, x_max, y_min, y_max = obstacle
            return [
                [(x_min, y_min), (x_max, y_min)],
                [(x_max, y_min), (x_max, y_max)],
                [(x_max, y_max), (x_min, y_max)],
                [(x_min, y_max), (x_min, y_min)],
            ]

        # Iterate over the obstacles in the map to find the closest distance (if there is one). Remember that the
        # obstacles are represented as a list of AABBs (Axis-Aligned Bounding Boxes) with the format
        # (x_min, x_max, y_min, y_max).
        result = None
        origin = np.array(origin)

        for obstacle in self.obstacles:
            for line in lines_from_obstacle(obstacle):
                p = ray_line_intersection(origin, angle, line[0], line[1])
                if p is None:
                    continue

                dist = np.linalg.norm(np.array(p) - origin)
                if result is None:
                    result = dist
                else:
                    result = min(result, dist)
        return result

# PID controller class
######### Your code starts here #########
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def control(self, error: float) -> float:
        """Simple PID control using internal timing - returns control signal."""
        now = time()
        if self.last_time is None:
            dt = 1e-3
        else:
            dt = max(1e-3, now - self.last_time)
        self.last_time = now
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
######### Your code ends here #########


class Particle:
    def __init__(self, x: float, y: float, theta: float, log_p: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.log_p = log_p

    def __str__(self) -> str:
        return f"Particle<pose: {self.x, self.y, self.theta}, log_p: {self.log_p}>"


class ParticleFilter:

    def __init__(
        self,
        map_: Map,
        n_particles: int,
        translation_variance: float,
        rotation_variance: float,
        measurement_variance: float,
    ):
        self.particles_visualization_pub = rospy.Publisher("/pf_particles", PoseArray, queue_size=10)
        self.estimate_visualization_pub = rospy.Publisher("/pf_estimate", PoseStamped, queue_size=10)

        # Initialize uniformly-distributed particles
        ######### Your code starts here #########
        # store map and parameters
        self.map_ = map_
        self.n_particles = n_particles
        self.translation_variance = translation_variance
        self.rotation_variance = rotation_variance
        self.measurement_variance = measurement_variance
        
        self.update_count = 0

        # initialize particles uniformly across map AABB
        x_min, x_max, y_min, y_max = map_.map_aabb
        self._particles: List[Particle] = []
        uniform_logp = math.log(1.0 / float(self.n_particles))
        for _ in range(self.n_particles):
            x = uniform(x_min, x_max)
            y = uniform(y_min, y_max)
            theta = uniform(-math.pi, math.pi)
            p = Particle(x, y, theta, uniform_logp)
            self._particles.append(p)
        ######### Your code ends here #########

    def visualize_particles(self):
        pa = PoseArray()
        pa.header.frame_id = "odom"
        pa.header.stamp = rospy.Time.now()
        for particle in self._particles:
            pose = Pose()
            pose.position = Point(particle.x, particle.y, 0.01)
            q_np = quaternion_from_euler(0, 0, float(particle.theta))
            pose.orientation = Quaternion(*q_np.tolist())
            pa.poses.append(pose)
        self.particles_visualization_pub.publish(pa)

    def visualize_estimate(self):
        ps = PoseStamped()
        ps.header.frame_id = "odom"
        ps.header.stamp = rospy.Time.now()
        x, y, theta = self.get_estimate()
        pose = Pose()
        pose.position = Point(x, y, 0.01)
        q_np = quaternion_from_euler(0, 0, float(theta))
        pose.orientation = Quaternion(*q_np.tolist())
        ps.pose = pose
        self.estimate_visualization_pub.publish(ps)

    def move_by(self, delta_x, delta_y, delta_theta):
        delta_theta = angle_to_neg_pi_to_pi(delta_theta)

        # Propagate motion of each particle
        ######### Your code starts here #########
        if abs(delta_x) < 1e-4 and abs(delta_y) < 1e-4 and abs(delta_theta) < 1e-4:
            return
        
        x_min, x_max, y_min, y_max = self.map_.map_aabb
        
        for p in self._particles:
            # add Gaussian noise to translation components
            dx_noisy = delta_x + np.random.normal(0.0, self.translation_variance)
            dy_noisy = delta_y + np.random.normal(0.0, self.translation_variance)
            # rotate noisy translation into world frame using particle heading
            world_dx = dx_noisy * math.cos(p.theta) - dy_noisy * math.sin(p.theta)
            world_dy = dx_noisy * math.sin(p.theta) + dy_noisy * math.cos(p.theta)
            p.x += world_dx
            p.y += world_dy

            # add Gaussian noise to rotation
            dtheta_noisy = delta_theta + np.random.normal(0.0, self.rotation_variance)
            p.theta = angle_to_neg_pi_to_pi(p.theta + dtheta_noisy)
            # keep log_p unchanged for motion (motion model assumed symmetric)
            
            if not (x_min <= p.x <= x_max and y_min <= p.y <= y_max):
                p.x = min(max(p.x, x_min), x_max)
                p.y = min(max(p.y, y_min), y_max)
                
            for (ox_min, ox_max, oy_min, oy_max) in self.map_.obstacles:
                if ox_min <= p.x <= ox_max and oy_min <= p.y <= oy_max:
                    p.x = uniform(x_min, x_max)
                    p.y = uniform(y_min, y_max)
                    p.theta = uniform(-math.pi, math.pi)
                    p.log_p = math.log(1.0 / float(self.n_particles))
        ######### Your code ends here #########

    def measure(self, z: float, scan_angle_in_rad: float):
        """Update the particles based on the measurement `z` at the given `scan_angle_in_rad`.

        Args:
            z: distance to an obstacle
            scan_angle_in_rad: Angle in the robots frame where the scan was taken
        """

        # Calculate posterior probabilities and resample
        ######### Your code starts here #########
        # accumulate log-likelihoods into log_p, then normalize and resample
        log_ps = []
        for p in self._particles:
            ray_angle_world = angle_to_neg_pi_to_pi(p.theta + scan_angle_in_rad)
            predicted = self.map_.closest_distance((p.x, p.y), ray_angle_world)
            if predicted is None:
                predicted = max(10.0, z + 5.0)
            try:
                log_likelihood = scipy.stats.norm(loc=predicted, scale=self.measurement_variance).logpdf(z)
            except Exception:
                prob = scipy.stats.norm(predicted, self.measurement_variance).pdf(z)
                log_likelihood = math.log(prob + 1e-12)
            # update particle log probability (Bayes: multiply prior by likelihood => add log)
            p.log_p = p.log_p + log_likelihood
            log_ps.append(p.log_p)

        # normalize log probabilities to get weights
        max_log = max(log_ps)
        shifted = [math.exp(lp - max_log) for lp in log_ps]
        total = sum(shifted)
        if total == 0 or math.isclose(total, 0.0):
            weights = np.ones(len(self._particles)) / float(len(self._particles))
        else:
            weights = np.array(shifted) / float(total)

        # set normalized log_p for each particle
        for p, w in zip(self._particles, weights):
            p.log_p = math.log(max(w, 1e-300))

        # resample according to weights
        do_resample = True
        neff = 1.0 / np.sum(weights ** 2)
        
        if self.update_count < 15:
            do_resample = False
        elif neff >= 0.8 * self.n_particles:
            do_resample = False
            
        self.update_count += 1
    
        if do_resample:
            indices = np.random.choice(
                range(len(self._particles)),
                size=len(self._particles),
                replace=True,
                p=weights,
            )
            new_particles = []
            for idx in indices:
                src = self._particles[idx]
                new_particles.append(
                    copy.deepcopy(
                        Particle(
                            src.x, src.y, src.theta,
                            math.log(1.0 / float(self.n_particles))
                        )
                    )
                )
            self._particles = new_particles
            
            for p in self._particles:
                p.x += np.random.normal(0.0, 0.01)
                p.y += np.random.normal(0.0, 0.01)
                p.theta = angle_to_neg_pi_to_pi(p.theta + np.random.normal(0.0, 0.01))
        ######### Your code ends here #########

    def get_estimate(self) -> Tuple[float, float, float]:
        # Estimate robot's location using particle weights
        ######### Your code starts here #########
        log_ps = np.array([p.log_p for p in self._particles])
        # stabilize then exponentiate
        max_log = np.max(log_ps)
        weights = np.exp(log_ps - max_log)
        s = np.sum(weights)
        if s == 0:
            weights = np.ones_like(weights) / float(len(weights))
        else:
            weights = weights / s

        xs = np.array([p.x for p in self._particles])
        ys = np.array([p.y for p in self._particles])
        thetas = np.array([p.theta for p in self._particles])

        x_est = float(np.sum(xs * weights))
        y_est = float(np.sum(ys * weights))
        sin_mean = float(np.sum(np.sin(thetas) * weights))
        cos_mean = float(np.sum(np.cos(thetas) * weights))
        theta_est = math.atan2(sin_mean, cos_mean)
        return x_est, y_est, theta_est
        ######### Your code ends here #########


class Controller:
    def __init__(self, particle_filter: ParticleFilter):
        rospy.init_node("particle_filter_controller", anonymous=True)
        self._particle_filter = particle_filter
        self._particle_filter.visualize_particles()

        #
        self.current_position = None
        self.last_odom = None
        self.laserscan = None
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.laserscan_sub = rospy.Subscriber("/scan", LaserScan, self.robot_laserscan_callback)
        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pointcloud_pub = rospy.Publisher("/scan_pointcloud", PointCloud, queue_size=10)
        self.target_position_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)

        while ((self.current_position is None) or (self.laserscan is None)) and (not rospy.is_shutdown()):
            rospy.loginfo("waiting for odom and laserscan")
            rospy.sleep(0.1)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        new_pose = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        # if we have a previous odom, compute delta and move PF
        if self.last_odom is not None:
            dx_world = new_pose["x"] - self.last_odom["x"]
            dy_world = new_pose["y"] - self.last_odom["y"]
            dtheta = angle_to_neg_pi_to_pi(new_pose["theta"] - self.last_odom["theta"])

            # convert world delta to robot frame of previous pose
            ct = math.cos(self.last_odom["theta"])
            st = math.sin(self.last_odom["theta"])
            dx_robot = ct * dx_world + st * dy_world
            dy_robot = -st * dx_world + ct * dy_world

            self._particle_filter.move_by(dx_robot, dy_robot, dtheta)

        self.last_odom = new_pose
        self.current_position = new_pose


    def robot_laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    def visualize_laserscan_ranges(self, idx_groups: List[Tuple[int, int]]):
        """Helper function to visualize ranges of sensor readings from the laserscan lidar.

        Example usage for visualizing the first 10 and last 10 degrees of the laserscan:
            `self.visualize_laserscan_ranges([(0, 10), (350, 360)])`
        """
        pcd = PointCloud()
        pcd.header.frame_id = "odom"
        pcd.header.stamp = rospy.Time.now()
        for idx_low, idx_high in idx_groups:
            for idx, d in enumerate(self.laserscan.ranges[idx_low:idx_high]):
                if d == inf:
                    continue
                angle = math.radians(idx) + self.current_position["theta"]
                x = d * math.cos(angle) + self.current_position["x"]
                y = d * math.sin(angle) + self.current_position["y"]
                z = 0.1
                pcd.points.append(Point32(x=x, y=y, z=z))
                pcd.channels.append(ChannelFloat32(name="rgb", values=(0.0, 1.0, 0.0)))
        self.pointcloud_pub.publish(pcd)

    def visualize_position(self, x: float, y: float):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = Point(x, y, 0.0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.scale = Vector3(0.075, 0.075, 0.1)
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
        marker_array.markers.append(marker)
        self.target_position_pub.publish(marker_array)

    def take_measurements(self):
        # Take measurement using LIDAR
        # NOTE: with more than 2 angles the particle filter will converge too quickly, so with high likelihood the
        # correct neighborhood won't be found.
        angle_min = self.laserscan.angle_min
        angle_increment = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        num_ranges = len(ranges)

        mid_idx = num_ranges // 2
        offset = int(15.0 / (angle_increment * 180.0 / math.pi))  # 15 degrees offset

        # Get indices for -15°, 0°, +15°
        indices = [max(0, min(num_ranges - 1, mid_idx + i)) for i in (-offset, 0, offset)]
        measurements = []

        for i in indices:
            z = ranges[i]
            if z == inf or np.isinf(z):
                if hasattr(self.laserscan, "range_max"):
                    z = self.laserscan.range_max
                else:
                    z = 10.0
            angle = angle_min + i * angle_increment
            measurements.append((z, angle))

        # Update PF with each beam
        for z, a in measurements:
            self._particle_filter.measure(z, a)
        ######### Your code ends here #########

    def autonomous_exploration(self):
        """Randomly explore the environment here, while making sure to call `take_measurements()` and
        `_particle_filter.move_by()`. The particle filter should converge on the robots position eventually.

        Note that the following visualizations functions are available:
            visualize_position(...)
            visualize_laserscan_ranges(...)
        """
        # Robot autonomously explores environment while it localizes itself
        ######### Your code starts here #########
        rate = rospy.Rate(1.0)  # explore at ~1 Hz loop
        max_steps = 400
        rotation_attempts = 0
        move_distance = 0.25  # move farther per step

        for step in range(max_steps):
            if rospy.is_shutdown():
                break
                
            # --- Prevent getting stuck spinning ---
            if rotation_attempts > 5:
                rospy.loginfo("Too many rotations; moving forward to escape.")
                self.forward_action(0.3)
                rotation_attempts = 0

            # Get front range safely
            front_range = None
            too_close = False

            if self.laserscan is not None:
                angle_min = self.laserscan.angle_min
                angle_inc = self.laserscan.angle_increment
                ranges = self.laserscan.ranges
                num_ranges = len(ranges)

                # --- FRONT WINDOW ONLY ---
                # we look at ~ +/- 25 degrees in front of robot
                front_window_deg = 25.0
                low_angle = -math.radians(front_window_deg)
                high_angle = math.radians(front_window_deg)

                low_idx = int(round((low_angle - angle_min) / angle_inc))
                high_idx = int(round((high_angle - angle_min) / angle_inc))
                low_idx = max(0, min(low_idx, num_ranges - 1))
                high_idx = max(0, min(high_idx, num_ranges - 1))
                if low_idx > high_idx:
                    low_idx, high_idx = high_idx, low_idx

                front_sector = [r for r in ranges[low_idx:high_idx + 1] if not np.isinf(r)]

                # also get the exact forward beam
                zero_idx = int(round((0.0 - angle_min) / angle_inc))
                zero_idx = max(0, min(zero_idx, num_ranges - 1))
                front_range = ranges[zero_idx]

                # decide "too close" based on this sector only
                if len(front_sector) > 0 and min(front_sector) < 0.28:
                    close_count += 1
                else:
                    close_count = 0

                # require it to be close twice in a row to react
                if close_count >= 2:
                    too_close = True

            if too_close:
                rospy.loginfo("Too close to obstacle, backing up & rotating.")
                self.forward_action(-0.12)
                self.rotate_action(uniform(math.pi / 5, math.pi / 3))  # bigger rotation away
                rotation_attempts += 1
                rate.sleep()
                continue

            # --- Main motion policy ---
            if front_range is None or np.isinf(front_range) or front_range > 0.7:
                # Move forward more confidently if clear
                self.forward_action(move_distance)
                rotation_attempts = 0
            else:
                rospy.loginfo("Obstacle ahead, rotating to find new direction.")
                self.rotate_action(uniform(math.pi / 4, math.pi / 2))
                rotation_attempts += 1

            # --- take PF measurements in a consistent way ---
            self.take_measurements()

            # --- visualize and check convergence ---
            self._particle_filter.visualize_particles()
            self._particle_filter.visualize_estimate()

            x_est, y_est, theta_est = self._particle_filter.get_estimate()
            pts = np.array([[p.x, p.y] for p in self._particle_filter._particles])
            if pts.shape[0] > 0:
                dists = np.linalg.norm(pts - np.array([x_est, y_est]), axis=1)
                std_dev = np.std(dists)
                rospy.loginfo(f"[Step {step}] Particle spread: {std_dev:.3f}")
                
                sensor_ok = False
                if front_range is not None and not np.isinf(front_range):
                    # predicted front range from PF estimate
                    predicted_front = self._particle_filter.map_.closest_distance(
                        (x_est, y_est), theta_est
                    )
                    if predicted_front is None:
                        predicted_front = 10.0
                    # if predicted and actual are close, we believe the pose
                    if abs(predicted_front - front_range) < 0.25:
                        sensor_ok = True

                if std_dev < 0.12 and sensor_ok:
                    rospy.loginfo("Particle filter converged (std < 0.12 and sensor matched).")
                    break

            rate.sleep()
        ######### Your code ends here #########

    def forward_action(self, distance: float):
        # Robot moves forward by a set amount during manual control
        ######### Your code starts here #########
        # publish a constant forward velocity for the duration needed to cover distance
        vel = Twist()
        speed = 0.15  # m/s
        vel.linear.x = float(speed if distance >= 0 else -speed)
        duration = abs(distance) / speed if speed > 0 else 0.0
        start = rospy.Time.now().to_sec()
        r = rospy.Rate(10)
        while (rospy.Time.now().to_sec() - start) < duration and (not rospy.is_shutdown()):
            self.robot_ctrl_pub.publish(vel)
            r.sleep()
        # stop
        vel.linear.x = 0.0
        self.robot_ctrl_pub.publish(vel)
        # PF will be updated in odom_callback()
        ######### Your code ends here #########

    def rotate_action(self, goal_theta: float):
        # Robot turns by a set amount during manual control
        ######### Your code starts here #########
        # goal_theta is a relative rotation (radians)
        vel = Twist()
        angular_speed = 0.8  # rad/s
        vel.angular.z = angular_speed if goal_theta >= 0 else -angular_speed
        duration = abs(goal_theta) / angular_speed if angular_speed > 0 else 0.0
        start = rospy.Time.now().to_sec()
        r = rospy.Rate(10)
        while (rospy.Time.now().to_sec() - start) < duration and (not rospy.is_shutdown()):
            self.robot_ctrl_pub.publish(vel)
            r.sleep()
        # stop rotating
        vel.angular.z = 0.0
        self.robot_ctrl_pub.publish(vel)
        # PF will be updated in odom_callback()
        ######### Your code ends here #########


""" Example usage

rosrun development lab8_9.py --map_filepath src/csci455l/scripts/lab8_9_map.json
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()
    with open(args.map_filepath, "r") as f:
        map_ = json.load(f)
        obstacles = map_["obstacles"]
        map_aabb = map_["map_aabb"]

    map_ = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.003
    rotation_variance = 0.03
    measurement_variance = 0.35
    particle_filter = ParticleFilter(map_, num_particles, translation_variance, rotation_variance, measurement_variance)
    controller = Controller(particle_filter)

    try:
        # Manual control
        goal_theta = 0
        controller.take_measurements()
        # while not rospy.is_shutdown():
        #     print("\nEnter 'a', 'w', 's', 'd' to move the robot:")
        #     uinput = input("")
        #     if uinput == "w": # forward
        #         ######### Your code starts here #########
        #         controller.forward_action(0.2)
        #         ######### Your code ends here #########
        #     elif uinput == "a": # left
        #         ######### Your code starts here #########
        #         controller.rotate_action(math.pi / 2.0)
        #         ######### Your code ends here #########
        #     elif uinput == "d": #right
        #         ######### Your code starts here #########
        #         controller.rotate_action(-math.pi / 2.0)
        #         ######### Your code ends here #########
        #     elif uinput == "s": # backwards
        #         ######### Your code starts here #########
        #         controller.forward_action(-0.2)
        #         ######### Your code ends here #########
        #     else:
        #         print("Invalid input")
        # #     ######### Your code starts here #########
        #     controller.take_measurements()
        #     # Visualize updated particles and estimate
        #     controller._particle_filter.visualize_particles()
        #     controller._particle_filter.visualize_estimate()
        # #     ######### Your code ends here #########

        # Autonomous exploration
        ######### Your code starts here #########
        for _ in range(1):
            controller.take_measurements()
            rospy.sleep(0.1)
        controller.autonomous_exploration()
        ######### Your code ends here #########

    except rospy.ROSInterruptException:
        print("Shutting down...")
