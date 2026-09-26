#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re as _re
import threading
import traceback
from typing import Any as _Any
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

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
        # Executor 오류 감지는 상태 출력과 별개로 유지한다.
        self._spin_error = None
        self._spin_exited = False

        # 출발 차선 및 ROS 2 모드 파라미터
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
        # Global path 읽기 및 경로 포인트 수 확인
        self.config_manager = ConfigManager(self)
        self.globalPathIn = self.config_manager.get_path('in')
        self.globalPathOut = self.config_manager.get_path('out')
        # Publish the selected paths immediately.  Path publication must not
        # depend on GPS/heading becoming valid first; RViz and other late
        # subscribers use the transient-local QoS to receive this snapshot.
        self.config_manager.pub_path()
        numGlobalPathIn = _get_parameter(self, '/mode/numGlobalPathIn')
        numGlobalPathOut = _get_parameter(self, '/mode/numGlobalPathOut')

        # 모듈 초기화
        self.logger = Logger(self)
        self.gps_sensor = GPS(self)
        self.lidar_sensor = LidarSensor(self)
        self.vehicle_state = VehicleState(self)
        self.path_planner = PathPlanner(self, self.globalPathIn, self.globalPathOut)
        self.speed_decision = SpeedDecision(self)
        self.steering_decision = SteeringDecision(self, numGlobalPathIn, numGlobalPathOut, self.globalPathIn, self.globalPathOut)
        self.line_changer = LineChanger(self)
        self.merge_checker = MergeChecker()
        self.vehicle_control = VehicleControler(self)

        # UI 모니터링 publishers
        self.path_idx_pub = self.create_publisher(Float32, _topic(self, '/curr_path_idx'), CONTROL_QOS)
        self.line_changing_pub = self.create_publisher(Bool, _topic(self, '/is_line_changing'), CONTROL_QOS)

        self.rate = self.create_rate(10)

    def _raise_if_spin_failed(self):
        """Turn a dead executor thread into an actionable main-thread error."""
        if self._spin_error is not None:
            raise RuntimeError(
                'executor.spin() terminated unexpectedly; '
                f'traceback follows:\n{self._spin_error}'
            )
        if self._spin_exited and rclpy.ok():
            raise RuntimeError(
                'executor.spin() returned while rclpy.ok() is still true'
            )

    def run(self):
        target_velocity = 0
        is_line_changing = False
        try:
            while rclpy.ok():
                self._raise_if_spin_failed()

                # 1. 센서 데이터 수집
                odom_back, heading, gps_status_back, gps_covariance_back, checkGPS_back, checkHeading, last_back_gps_in = self.gps_sensor.gps_data_back()
                odom_front, gps_status_front, gps_covariance_front, checkGPS_front, last_front_gps_in = self.gps_sensor.gps_data_front()
                velocity, _, checkERP = self.vehicle_state.vehicle_data()
                # Localization must remain visible while the controller waits
                # for ERP42 feedback. Never turn an old/no-fix pose into a
                # current transform, and keep all driving gates below intact.
                gps_age = (self.get_clock().now() - last_back_gps_in).nanoseconds / 1e9
                if checkGPS_back and checkHeading and gps_status_back >= 0 and 0.0 <= gps_age <= 1.0:
                    self.config_manager.broadcast(odom_back)
                if (checkGPS_front or checkGPS_back) and checkHeading and checkERP:
                    lidar_front, lidar_side_front, lidar_side_back = self.lidar_sensor.lidar_data(self.lane)

                    # 2. 위치 파악 및 현재 경로 인덱스 발행
                    curWaypointIndxIn, curWaypointIndxOut, distanceIn, distanceOut = self.path_planner.localization(odom_back, self.lane)
                    in_path_idx = self.globalPathIn.poses[curWaypointIndxIn].pose.orientation.w
                    out_path_idx = self.globalPathOut.poses[curWaypointIndxOut].pose.orientation.w
                    curr_path_idx, distance = (in_path_idx, distanceIn) if self.lane == 1 else (out_path_idx, distanceOut)
                    self.path_idx_pub.publish(Float32(data=curr_path_idx))

                    # 3. GPS 데이터 유효성 확인
                    gps_error_status = self.speed_decision.isGPSerror(gps_status_back, gps_covariance_back, last_back_gps_in, gps_status_front, gps_covariance_front, last_front_gps_in, curr_path_idx)
                    if gps_error_status == 'both_error':
                        self.get_logger().warning('Both GPS Error. Stop.')
                        continue
                    elif gps_error_status == 'back_error':
                        self.get_logger().warning('Back GPS Error. Using Front GPS.')
                        continue
                    elif gps_error_status == 'front_error':
                        self.get_logger().warning('Front GPS Error. Using Back GPS.')

                    # 4. 병합 구간 확인 및 차선 변경
                    velocity_scale_side = self.merge_checker.merge_check(curr_path_idx, lidar_side_front, lidar_side_back)
                    change, _, is_line_changing = self.line_changer.line_change_check(curr_path_idx, distance, target_velocity, lidar_front, lidar_side_front, lidar_side_back, self.lane)
                    if change:
                        self.lane = 1 if self.lane == 2 else 2
                    self.line_changing_pub.publish(Bool(data=is_line_changing))

                    # 5. 목표 속도 및 조향각 결정
                    if curr_path_idx == 1 or curr_path_idx == 2:
                        target_velocity, brake = self.speed_decision.control_speed_curve(lidar_front, curr_path_idx, velocity_scale_side)
                    else:
                        target_velocity, brake = self.speed_decision.control_speed(lidar_front, curr_path_idx, velocity_scale_side)
                    if is_line_changing:
                        target_velocity = min(target_velocity, 20)
                    steering_angle = self.steering_decision.purePursuit(odom_back, self.lane, velocity, heading, curr_path_idx, curWaypointIndxIn, curWaypointIndxOut, is_line_changing)

                    # 6. 차량 제어 명령 실행
                    self.vehicle_control.control(target_velocity, brake, steering_angle)
                self.rate.sleep()
        except Exception:
            if not rclpy.ok():
                return
            raise


def main(args=None):
    node = None
    executor = None
    spin_thread = None
    try:
        rclpy.init(args=args)
        node = AutonomousVehicleController()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)

        def _spin_executor():
            try:
                executor.spin()
                node._spin_exited = True
            except BaseException:
                node._spin_error = traceback.format_exc()
                raise

        spin_thread = threading.Thread(
            target=_spin_executor,
            name='planner_vector_executor',
            daemon=True,
        )
        spin_thread.start()
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception:
        if node is not None:
            node.get_logger().fatal(
                'planner_vector main() failed:\n'
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
