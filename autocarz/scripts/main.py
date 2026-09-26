#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re as _re
import threading
import time
import traceback
from typing import Any as _Any
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data

CONTROL_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
_MISSING = object()

def _normalize_parameter_name(name: str) -> str:
    value = str(name).strip()
    if value.startswith('~'):
        value = value[1:]
    value = value.strip('/').replace('/', '.')
    value = _re.sub(r'[^A-Za-z0-9_.]', '_', value)
    value = _re.sub(r'\.+', '.', value).strip('.')
    return value or 'unnamed'

def _get_parameter(node: Node, name: str, default: _Any = _MISSING):
    parameter_name = _normalize_parameter_name(name)
    if not node.has_parameter(parameter_name):
        if default is _MISSING:
            raise KeyError(f"필수 ROS 2 파라미터 '{parameter_name}'가 없습니다.")
        node.declare_parameter(parameter_name, default)
    return node.get_parameter(parameter_name).value

def _topic(node: Node, default_topic: str) -> str:
    slug = default_topic.strip('/')
    slug = _re.sub(r'[^A-Za-z0-9_]', '_', slug)
    slug = _re.sub(r'_+', '_', slug).strip('_') or 'root'
    parameter_name = f'topics.{slug}'
    if not node.has_parameter(parameter_name):
        node.declare_parameter(parameter_name, default_topic)
    return str(node.get_parameter(parameter_name).value)

from sensor_data import LidarSensor, GPS, VehicleState
from online import PathPlanner, SpeedDecision, SteeringDecision, LineChanger, MergeChecker
from vehicle_control import VehicleControler
from utilities import Logger, ConfigManager
from std_msgs.msg import Float32, Bool

class AutonomousVehicleController(Node):

    def __init__(self):
        super().__init__('planner_vector')
        # Debug instrumentation is enabled by default so that a launch-time
        # planner_node failure leaves a useful reason in the ROS log.  It only
        # observes state and does not change the control decisions below.
        self.declare_parameter('debug.enabled', True)
        self.declare_parameter('debug.report_period_sec', 2.0)
        debug_value = self.get_parameter('debug.enabled').value
        if isinstance(debug_value, str):
            self._debug_enabled = debug_value.strip().lower() in (
                '1', 'true', 'yes', 'on'
            )
        else:
            self._debug_enabled = bool(debug_value)
        try:
            self._debug_report_period = max(
                0.5,
                float(self.get_parameter('debug.report_period_sec').value),
            )
        except (TypeError, ValueError):
            self._debug_report_period = 2.0
        self._debug_started_at = time.monotonic()
        self._debug_last_report_at = 0.0
        self._debug_loop_count = 0
        self._debug_active_loop_count = 0
        self._debug_last_step = 'constructor'
        self._debug_last_gate = 'not evaluated'
        self._debug_last_flags = {}
        self._debug_gate_counts = {
            'gps': 0,
            'heading': 0,
            'erp': 0,
            'ready': 0,
        }
        self._debug_spin_error = None
        self._debug_spin_exited = False
        self._debug_log('planner_vector constructor started')
        self.lane = 1
        # ROS 1의 private mode 파라미터를 ROS 2 선언 파라미터로 유지한다.
        # launch 없이 `ros2 run autocarz main.py`를 실행해도 final 경로가
        # 선택되도록 기본값을 제공하고, launch override는 그대로 우선한다.
        self.declare_parameter('mode.name', 'final')
        self.declare_parameter('mode.pathIn', 'path/UOS_test/new_path.txt')
        self.declare_parameter('mode.pathOut', 'path/UOS_test/new_path.txt')
        self.declare_parameter(
            'mode.velocity_scale_base',
            '{"0": 20.0, "1": 18.0, "2": 18.0, "3": 20.0, "4": 20.0}',
        )
        self.config_manager = self._debug_call(
            'ConfigManager()', lambda: ConfigManager(self)
        )
        self.globalPathIn = self._debug_call(
            'load path in', lambda: self.config_manager.get_path('in')
        )
        self.globalPathOut = self._debug_call(
            'load path out', lambda: self.config_manager.get_path('out')
        )
        # Publish the selected paths immediately.  Path publication must not
        # depend on GPS/heading becoming valid first; RViz and other late
        # subscribers use the transient-local QoS to receive this snapshot.
        self._debug_call('initial path publish', self.config_manager.pub_path)
        numGlobalPathIn = _get_parameter(self, '/mode/numGlobalPathIn')
        numGlobalPathOut = _get_parameter(self, '/mode/numGlobalPathOut')
        self.logger = self._debug_call('Logger()', lambda: Logger(self))
        self.gps_sensor = self._debug_call('GPS()', lambda: GPS(self))
        self.lidar_sensor = self._debug_call('LidarSensor()', lambda: LidarSensor(self))
        self.vehicle_state = self._debug_call('VehicleState()', lambda: VehicleState(self))
        self.path_planner = self._debug_call(
            'PathPlanner()',
            lambda: PathPlanner(self, self.globalPathIn, self.globalPathOut),
        )
        self.speed_decision = self._debug_call(
            'SpeedDecision()', lambda: SpeedDecision(self)
        )
        self.steering_decision = self._debug_call(
            'SteeringDecision()',
            lambda: SteeringDecision(
                self,
                numGlobalPathIn,
                numGlobalPathOut,
                self.globalPathIn,
                self.globalPathOut,
            ),
        )
        self.line_changer = self._debug_call(
            'LineChanger()', lambda: LineChanger(self)
        )
        self.merge_checker = self._debug_call(
            'MergeChecker()', lambda: MergeChecker()
        )
        self.vehicle_control = self._debug_call(
            'VehicleControler()', lambda: VehicleControler(self)
        )
        self.path_idx_pub = self._debug_call(
            'path index publisher',
            lambda: self.create_publisher(
                Float32, _topic(self, '/curr_path_idx'), CONTROL_QOS
            ),
        )
        self.line_changing_pub = self._debug_call(
            'line changing publisher',
            lambda: self.create_publisher(
                Bool, _topic(self, '/is_line_changing'), CONTROL_QOS
            ),
        )
        self.rate = self._debug_call('control rate', lambda: self.create_rate(10))
        self._debug_last_step = 'constructor complete'
        self._debug_log(
            '[DEBUG] initialization complete: '
            f'mode={self.get_parameter("mode.name").value}, '
            f'path_in_points={len(self.globalPathIn.poses)}, '
            f'path_out_points={len(self.globalPathOut.poses)}, '
            f'path_in={self.get_parameter("mode.pathIn").value}, '
            f'path_out={self.get_parameter("mode.pathOut").value}'
        )
        self._debug_report(force=True)

    def _debug_log(self, message, level='info'):
        """Write a debug message without affecting runtime behavior."""
        if not self._debug_enabled:
            return
        logger = self.get_logger()
        if level == 'fatal':
            logger.fatal(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        else:
            logger.info(message)

    def _debug_call(self, name, callback):
        """Log a constructor/setup step and its full exception traceback."""
        self._debug_last_step = f'initializing {name}'
        self._debug_log(f'[DEBUG] {name} started')
        try:
            result = callback()
        except Exception:
            self._debug_log(
                f'[DEBUG] {name} failed:\n{traceback.format_exc()}',
                level='fatal',
            )
            raise
        self._debug_log(f'[DEBUG] {name} complete')
        return result

    def _debug_report(self, force=False):
        """Periodically report input availability and the last control step."""
        if not self._debug_enabled:
            return
        now = time.monotonic()
        if not force and now - self._debug_last_report_at < self._debug_report_period:
            return
        self._debug_last_report_at = now

        flags = self._debug_last_flags
        flag_text = (
            f"gps_back={flags.get('gps_back', '?')}, "
            f"gps_front={flags.get('gps_front', '?')}, "
            f"heading={flags.get('heading', '?')}, "
            f"erp_raw={flags.get('erp_raw', '?')}"
        )
        topic_names = (
            '/ublox1/fix',
            '/ublox2/fix',
            '/erp42/state',
            '/waypointInfo',
            '/vlp/distance_front',
            '/livox/distance_front',
            '/livox/max_roi_dis_in',
            '/livox/max_roi_dis_out',
        )
        links = []
        for topic in topic_names:
            try:
                links.append(
                    f"{topic}:pub={self.count_publishers(topic)},"
                    f"sub={self.count_subscribers(topic)}"
                )
            except Exception as exc:
                links.append(f"{topic}:graph_error={type(exc).__name__}")

        self._debug_log(
            '[DEBUG] '
            f'uptime={now - self._debug_started_at:.1f}s, '
            f'loops={self._debug_loop_count}, '
            f'active_loops={self._debug_active_loop_count}, '
            f'gate={self._debug_last_gate}, '
            f'last_step={self._debug_last_step}, '
            f'flags=({flag_text}), '
            f'gate_counts={self._debug_gate_counts}, '
            f'links=[{"; ".join(links)}]'
        )

    def _debug_raise_if_spin_failed(self):
        """Turn a dead executor thread into an actionable main-thread error."""
        if self._debug_spin_error is not None:
            raise RuntimeError(
                'executor.spin() terminated unexpectedly; '
                f'traceback follows:\n{self._debug_spin_error}'
            )
        if self._debug_spin_exited and rclpy.ok():
            raise RuntimeError(
                'executor.spin() returned while rclpy.ok() is still true'
            )

    def run(self):
        target_velocity = 0
        is_line_changing = False
        self._debug_log('[DEBUG] run() entered')
        try:
            while rclpy.ok():
                self._debug_raise_if_spin_failed()
                self._debug_loop_count += 1
                self._debug_last_step = 'gps_data_back'
                odom_back, heading, gps_status_back, gps_covariance_back, checkGPS_back, checkHeading, last_back_gps_in = self.gps_sensor.gps_data_back()
                self._debug_last_step = 'gps_data_front'
                odom_front, gps_status_front, gps_covariance_front, checkGPS_front, last_front_gps_in = self.gps_sensor.gps_data_front()
                self._debug_last_step = 'vehicle_data'
                velocity, _, checkERP = self.vehicle_state.vehicle_data()
                self._debug_last_flags = {
                    'gps_back': checkGPS_back,
                    'gps_front': checkGPS_front,
                    'heading': checkHeading,
                    'erp_raw': checkERP,
                }
                # Localization must remain visible while the controller waits
                # for ERP42 feedback. Never turn an old/no-fix pose into a
                # current transform, and keep all driving gates below intact.
                gps_age = (self.get_clock().now() - last_back_gps_in).nanoseconds / 1e9
                if checkGPS_back and checkHeading and gps_status_back >= 0 and 0.0 <= gps_age <= 1.0:
                    self._debug_last_step = 'TF broadcast'
                    self.config_manager.broadcast(odom_back)
                if not (checkGPS_front or checkGPS_back):
                    self._debug_gate_counts['gps'] += 1
                    self._debug_last_gate = 'blocked: no GPS fix'
                elif not checkHeading:
                    self._debug_gate_counts['heading'] += 1
                    self._debug_last_gate = 'blocked: heading unavailable'
                elif not checkERP:
                    self._debug_gate_counts['erp'] += 1
                    self._debug_last_gate = 'blocked: vehicle state unavailable'
                else:
                    self._debug_gate_counts['ready'] += 1
                    self._debug_active_loop_count += 1
                    self._debug_last_gate = 'control path entered'
                    self._debug_last_step = 'lidar_data'
                    lidar_front, lidar_side_front, lidar_side_back = self.lidar_sensor.lidar_data(self.lane)
                    self._debug_last_step = 'path_planner.localization'
                    curWaypointIndxIn, curWaypointIndxOut, distanceIn, distanceOut = self.path_planner.localization(odom_back, self.lane)
                    self._debug_last_step = 'path index lookup'
                    in_path_idx = self.globalPathIn.poses[curWaypointIndxIn].pose.orientation.w
                    out_path_idx = self.globalPathOut.poses[curWaypointIndxOut].pose.orientation.w
                    curr_path_idx, distance = (in_path_idx, distanceIn) if self.lane == 1 else (out_path_idx, distanceOut)
                    self.path_idx_pub.publish(Float32(data=curr_path_idx))
                    self._debug_last_step = 'GPS error check'
                    gps_error_status = self.speed_decision.isGPSerror(gps_status_back, gps_covariance_back, last_back_gps_in, gps_status_front, gps_covariance_front, last_front_gps_in, curr_path_idx)
                    if gps_error_status == 'both_error':
                        self._debug_last_gate = 'stopped: both GPS error'
                        self.get_logger().warning('Both GPS Error. Stop.')
                        self._debug_report()
                        continue
                    elif gps_error_status == 'back_error':
                        self._debug_last_gate = 'stopped: back GPS error'
                        self.get_logger().warning('Back GPS Error. Using Front GPS.')
                        self._debug_report()
                        continue
                    elif gps_error_status == 'front_error':
                        self._debug_last_gate = 'warning: front GPS error'
                        self.get_logger().warning('Front GPS Error. Using Back GPS.')
                    self._debug_last_step = 'merge_check'
                    velocity_scale_side = self.merge_checker.merge_check(curr_path_idx, lidar_side_front, lidar_side_back)
                    self._debug_last_step = 'line_change_check'
                    change, _, is_line_changing = self.line_changer.line_change_check(curr_path_idx, distance, target_velocity, lidar_front, lidar_side_front, lidar_side_back, self.lane)
                    if change:
                        self.lane = 1 if self.lane == 2 else 2
                    self.line_changing_pub.publish(Bool(data=is_line_changing))
                    self._debug_last_step = 'speed decision'
                    if curr_path_idx == 1 or curr_path_idx == 2:
                        target_velocity, brake = self.speed_decision.control_speed_curve(lidar_front, curr_path_idx, velocity_scale_side)
                    else:
                        target_velocity, brake = self.speed_decision.control_speed(lidar_front, curr_path_idx, velocity_scale_side)
                    if is_line_changing:
                        target_velocity = min(target_velocity, 20)
                    self._debug_last_step = 'purePursuit'
                    steering_angle = self.steering_decision.purePursuit(odom_back, self.lane, velocity, heading, curr_path_idx, curWaypointIndxIn, curWaypointIndxOut, is_line_changing)
                    print(steering_angle)
                    self._debug_last_step = 'vehicle_control.control'
                    self.vehicle_control.control(target_velocity, brake, steering_angle)
                self._debug_report()
                self.rate.sleep()
        except Exception:
            if not rclpy.ok():
                self._debug_log(
                    '[DEBUG] run() interrupted by ROS shutdown'
                )
                return
            self._debug_log(
                '[DEBUG] run() terminated unexpectedly. '
                f'last_step={self._debug_last_step}, '
                f'last_gate={self._debug_last_gate}\n{traceback.format_exc()}',
                level='fatal'
            )
            raise
        finally:
            self._debug_report(force=True)
            self._debug_log('[DEBUG] run() leaving')

def main(args=None):
    node = None
    executor = None
    spin_thread = None
    try:
        rclpy.init(args=args)
        print('[DEBUG] rclpy.init() complete', flush=True)
        node = AutonomousVehicleController()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)

        def _spin_executor():
            try:
                node.get_logger().info('[DEBUG] executor.spin() entered')
                executor.spin()
                node._debug_spin_exited = True
                if rclpy.ok():
                    node.get_logger().error(
                        '[DEBUG] executor.spin() returned while rclpy.ok() is true'
                    )
            except BaseException:
                node._debug_spin_error = traceback.format_exc()
                node.get_logger().fatal(
                    '[DEBUG] executor thread terminated with an exception:\n'
                    f'{node._debug_spin_error}'
                )
                raise

        spin_thread = threading.Thread(
            target=_spin_executor,
            name='planner_vector_executor',
            daemon=True,
        )
        spin_thread.start()
        node.get_logger().info('[DEBUG] executor thread started')
        node.run()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info('[DEBUG] KeyboardInterrupt received')
    except Exception:
        if node is None:
            print('[DEBUG] planner_vector failed before node construction', flush=True)
            traceback.print_exc()
        else:
            node.get_logger().fatal(
                '[DEBUG] planner_vector main() failed:\n'
                f'{traceback.format_exc()}'
            )
        raise
    finally:
        try:
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            if spin_thread is not None:
                spin_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            # SIGINT may arrive during executor/node cleanup.
            pass

if __name__ == '__main__':
    main()
