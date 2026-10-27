#!/usr/bin/env python3
"""Student template for the holonomic obstacle wall follower."""

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan


def euler_from_quaternion(q):
    """Calcula roll, pitch, yaw de forma nativa sin dependencias rotas de NumPy."""
    if hasattr(q, 'x'):
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
    else:
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return 0.0, 0.0, yaw


def clamp(value, low, high):
    return max(low, min(high, value))


class HolonomicObstacleWallFollower(Node):
    """Complete the LiDAR processing, state machine and control laws."""

    def __init__(self):
        super().__init__('holonomic_obstacle_wall_follower')

        # Desired distance between the robot and the right wall (m).
        self.follow_distance = float(
            self.declare_parameter('follow_distance', 0.35).value
        )
        # Maximum lateral distance used to classify a short obstacle (m).
        self.max_front_length = float(
            self.declare_parameter('max_front_length', 0.50).value
        )
        # Forward velocity (m/s).
        self.forward_speed = float(
            self.declare_parameter('forward_speed', 0.20).value
        )
        # Maximum lateral velocity (m/s).
        self.lateral_speed = float(
            self.declare_parameter('lateral_speed', 0.20).value
        )
        # Turn speed and maximum angular correction (rad/s).
        self.turn_speed = float(
            self.declare_parameter('turn_speed', 1.00).value
        )
        # Maximum LiDAR distance considered (m).
        self.detection_limit = float(
            self.declare_parameter('detection_limit', 1.20).value
        )
        # Proportional gain for lateral-distance correction.
        self.lateral_gain = float(
            self.declare_parameter('lateral_gain', 2.0).value
        )
        # Proportional gain for angular alignment.
        self.alignment_gain = float(
            self.declare_parameter('alignment_gain', 2.0).value
        )

        self.RIGHT = 'RIGHT'
        self.FRONT = 'FRONT'
        self.FRONT_RIGHT = 'FRONT_RIGHT'
        self.BACK_RIGHT = 'BACK_RIGHT'

        self.diagonal_factor = 0.5
        self.region_confirmations = 3
        self.robust_rays = 3

        self.state = self.RIGHT
        self.obstacle_is_long = False
        self.passing_short_obstacle = False
        self.command = Twist()
        self.odom_pose = None
        self.front_start_pose = None
        self.right_angle = None

        self.region_candidate = None
        self.region_candidate_count = 0
        self.stable_region = None

        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.05, self.publish_command)

        self.get_logger().info(
            'Holonomic wall follower ready.'
        )

    def publish_command(self):
        self.publisher.publish(self.command)

    def odom_callback(self, msg):
        """Store planar pose (x, y, yaw) from odometry."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        quaternion = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]
        _, _, yaw = euler_from_quaternion(quaternion)
        self.odom_pose = (position.x, position.y, yaw)

    def region_measurements(self, scan):
        """Return robust distances and angles for the four LiDAR regions."""
        regions_data = {
            self.FRONT: [],
            self.FRONT_RIGHT: [],
            self.RIGHT: [],
            self.BACK_RIGHT: []
        }

        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min <= r <= scan.range_max and r < self.detection_limit:
                deg = math.degrees(math.atan2(math.sin(angle), math.cos(angle)))
                if -20.0 <= deg <= 20.0:
                    regions_data[self.FRONT].append((r, deg))
                elif -70.0 <= deg < -20.0:
                    regions_data[self.FRONT_RIGHT].append((r, deg))
                elif -110.0 <= deg < -70.0:
                    regions_data[self.RIGHT].append((r, deg))
                elif -160.0 <= deg < -110.0:
                    regions_data[self.BACK_RIGHT].append((r, deg))
            angle += scan.angle_increment

        results = {}
        for reg, rays in regions_data.items():
            if not rays:
                results[reg] = (math.inf, None)
            else:
                rays.sort(key=lambda x: x[0])
                k_rays = rays[:self.robust_rays]
                mean_dist = sum(d for d, _ in k_rays) / len(k_rays)
                mean_angle = sum(a for _, a in k_rays) / len(k_rays)
                results[reg] = (mean_dist, mean_angle)

        return results

    def closest_region(self, measurements):
        """Return the closest region inside detection_limit."""
        best_region = None
        min_dist = self.detection_limit
        for reg in [self.FRONT, self.FRONT_RIGHT, self.RIGHT, self.BACK_RIGHT]:
            dist, _ = measurements[reg]
            if dist < min_dist:
                min_dist = dist
                best_region = reg
        return best_region

    def confirmed_region(self, measurements):
        """Confirm a region change over several consecutive scans."""
        candidate = self.closest_region(measurements)
        if candidate == self.region_candidate:
            self.region_candidate_count += 1
        else:
            self.region_candidate = candidate
            self.region_candidate_count = 1

        if self.region_candidate_count >= self.region_confirmations:
            self.stable_region = self.region_candidate

        return self.stable_region

    def lateral_progress(self):
        """Return travel along the robot's initial lateral axis."""
        if self.odom_pose is None or self.front_start_pose is None:
            return 0.0
        x, y, _ = self.odom_pose
        x0, y0, yaw0 = self.front_start_pose
        dx = x - x0
        dy = y - y0
        return abs(-math.sin(yaw0) * dx + math.cos(yaw0) * dy)

    def follow_right(self, measurements, closest, cmd):
        """Follow the wall using lateral and angular correction."""
        front_dist, _ = measurements[self.FRONT]
        front_threshold = 1.5 * self.follow_distance

        if front_dist < front_threshold or closest == self.FRONT:
            if self.odom_pose is None:
                return  # sense /odom no s'inicia la maniobra
            self.state = self.FRONT
            self.front_start_pose = self.odom_pose
            self.passing_short_obstacle = False
            return

        right_dist, right_angle = measurements[self.RIGHT]
        cmd.linear.x = self.forward_speed

        if math.isfinite(right_dist):
            error_dist = right_dist - self.follow_distance
            cmd.linear.y = clamp(-self.lateral_gain * error_dist, -self.lateral_speed, self.lateral_speed)

            if not self.passing_short_obstacle and right_angle is not None:
                error_angle = math.radians(right_angle - (-90.0))
                cmd.angular.z = clamp(self.alignment_gain * error_angle, -self.turn_speed, self.turn_speed)
            else:
                cmd.angular.z = 0.0
        else:
            cmd.linear.y = -0.5 * self.lateral_speed
            cmd.angular.z = 0.0

    def follow_front(self, measurements, closest, cmd):
        """Move laterally and classify the obstacle as short or long."""
        front_dist, _ = measurements[self.FRONT]
        front_threshold = 1.5 * self.follow_distance

        if (not self.obstacle_is_long and front_dist >= front_threshold
                and closest != self.FRONT):
            self.state = self.FRONT_RIGHT
            self.passing_short_obstacle = True
            return

        if self.lateral_progress() >= self.max_front_length:
            self.obstacle_is_long = True
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = self.turn_speed
            if closest == self.RIGHT:
                self.state = self.RIGHT
                self.obstacle_is_long = False
            return

        cmd.linear.x = 0.0
        cmd.linear.y = self.lateral_speed

        if math.isfinite(front_dist):
            error_dist = front_dist - self.follow_distance
            cmd.linear.x = clamp(self.lateral_gain * error_dist, -self.forward_speed, self.forward_speed)

    def follow_front_right(self, measurements, closest, cmd):
        """Move diagonally around the front-right side."""
        if closest == self.RIGHT:
            self.state = self.RIGHT
            return

        cmd.linear.x = self.forward_speed * self.diagonal_factor
        cmd.linear.y = self.lateral_speed * self.diagonal_factor
        cmd.angular.z = 0.0

    def follow_back_right(self, measurements, closest, cmd):
        """Complete the bypass and recover the right wall."""
        if closest == self.RIGHT or measurements[self.RIGHT][0] < self.detection_limit:
            self.state = self.RIGHT
            self.passing_short_obstacle = False
            return

        cmd.linear.x = self.forward_speed * self.diagonal_factor
        cmd.linear.y = -self.lateral_speed * self.diagonal_factor
        cmd.angular.z = 0.0

    def scan_callback(self, scan):
        """Read the LiDAR and calculate the next velocity command."""
        measurements = self.region_measurements(scan)
        closest = self.confirmed_region(measurements)

        cmd = Twist()

        if self.state == self.RIGHT:
            self.follow_right(measurements, closest, cmd)
        elif self.state == self.FRONT:
            self.follow_front(measurements, closest, cmd)
        elif self.state == self.FRONT_RIGHT:
            self.follow_front_right(measurements, closest, cmd)
        elif self.state == self.BACK_RIGHT:
            self.follow_back_right(measurements, closest, cmd)

        self.command = cmd

    def stop(self):
        """Publish a final zero command."""
        self.command = Twist()
        self.timer.cancel()
        try:
            self.publisher.publish(self.command)
        except Exception:
            pass


def main(args=None):
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = HolonomicObstacleWallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()