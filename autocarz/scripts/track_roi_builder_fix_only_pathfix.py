#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 Dual u-blox Track Line Recorder

기본 GPS Topic:
    /ublox1/fix

전방 GPS 사용 시:
    --topic /ublox2/fix

저장 위치:
    autocarz/path/track_data/

생성 파일:
    left_raw.csv
    middle_raw.csv
    middle1_raw.csv
    middle2_raw.csv
    right_raw.csv

사용 예시:

1) MIDDLE 기록
python3 track_roi_recorder_ublox.py record --line middle

2) LEFT 기록
python3 track_roi_recorder_ublox.py record --line left

3) RIGHT 기록
python3 track_roi_recorder_ublox.py record --line right

4) ublox2 사용
python3 track_roi_recorder_ublox.py record \
    --line middle \
    --topic /ublox2/fix

5) GPS 확인
ros2 topic hz /ublox1/fix
ros2 topic echo /ublox1/fix --once
"""

import argparse
import csv
import math
import os
import sys
from typing import Optional, Tuple


# ============================================================
# 기본 경로
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PACKAGE_DIR = os.path.dirname(
    SCRIPT_DIR
)

PATH_DIR = os.path.join(
    PACKAGE_DIR,
    "path",
)

DEFAULT_DATA_DIR = os.path.join(
    PATH_DIR,
    "track_data",
)


# ============================================================
# GPS Recorder
# ============================================================

class UbloxTrackRecorder:
    def __init__(
        self,
        line_name: str,
        output_path: str,
        topic: str,
        min_distance: float,
        map_offset_x: float,
        map_offset_y: float,
    ):
        self.line_name = str(
            line_name
        )

        self.output_path = os.path.abspath(
            output_path
        )

        self.topic = str(
            topic
        )

        self.min_distance = float(
            min_distance
        )

        self.map_offset_x = float(
            map_offset_x
        )

        self.map_offset_y = float(
            map_offset_y
        )

        if self.min_distance < 0.0:
            raise ValueError(
                "--min-distance는 0 이상이어야 합니다."
            )

        self.last_xy: Optional[
            Tuple[float, float]
        ] = None

        self.count = 0

        self.node = None
        self.subscription = None

        self.fp = None
        self.writer = None

        self.closed_file = False

        self.first_message_received = False


    # ========================================================
    # GPS -> XY
    # ========================================================

    @staticmethod
    def latlong2xy(
        latitude: float,
        longitude: float,
        map_offset_x: float,
        map_offset_y: float,
    ) -> Tuple[
        float,
        float,
    ]:
        """
        기존 autocarz 좌표 변환 방식.

        NavSatFix:
            latitude
            longitude

        출력:
            local x [m]
            local y [m]
        """

        RE = 6371.00877
        GRID = 5.0e-7

        SLAT1 = 30.0
        SLAT2 = 60.0

        OLON = 126.0
        OLAT = 38.0

        XO = 43.0
        YO = 136.0

        DEGRAD = math.pi / 180.0

        re = RE / GRID

        slat1 = (
            SLAT1
            *
            DEGRAD
        )

        slat2 = (
            SLAT2
            *
            DEGRAD
        )

        olon = (
            OLON
            *
            DEGRAD
        )

        olat = (
            OLAT
            *
            DEGRAD
        )

        sn = (
            math.tan(
                math.pi * 0.25
                +
                slat2 * 0.5
            )
            /
            math.tan(
                math.pi * 0.25
                +
                slat1 * 0.5
            )
        )

        sn = (
            math.log(
                math.cos(slat1)
                /
                math.cos(slat2)
            )
            /
            math.log(sn)
        )

        sf = math.tan(
            math.pi * 0.25
            +
            slat1 * 0.5
        )

        sf = (
            math.pow(
                sf,
                sn,
            )
            *
            math.cos(slat1)
            /
            sn
        )

        ro = math.tan(
            math.pi * 0.25
            +
            olat * 0.5
        )

        ro = (
            re
            *
            sf
            /
            math.pow(
                ro,
                sn,
            )
        )

        ra = math.tan(
            math.pi * 0.25
            +
            latitude
            *
            DEGRAD
            *
            0.5
        )

        ra = (
            re
            *
            sf
            /
            math.pow(
                ra,
                sn,
            )
        )

        theta = (
            longitude
            *
            DEGRAD
            -
            olon
        )

        if theta > math.pi:
            theta -= (
                2.0
                *
                math.pi
            )

        if theta < -math.pi:
            theta += (
                2.0
                *
                math.pi
            )

        theta *= sn

        rs_x = math.floor(
            ra
            *
            math.sin(theta)
            +
            XO
            +
            0.5
        )

        rs_y = math.floor(
            ro
            -
            ra
            *
            math.cos(theta)
            +
            YO
            +
            0.5
        )

        x = (
            rs_x
            /
            2000.0
            +
            map_offset_x
        )

        y = (
            rs_y
            /
            2000.0
            +
            map_offset_y
        )

        return (
            float(x),
            float(y),
        )


    # ========================================================
    # CSV
    # ========================================================

    def open_output_file(
        self,
    ):
        output_dir = os.path.dirname(
            self.output_path
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        self.fp = open(
            self.output_path,
            "w",
            newline="",
            buffering=1,
            encoding="utf-8",
        )

        self.writer = csv.writer(
            self.fp
        )

        self.writer.writerow(
            [
                "x",
                "y",
                "z",
                "stamp",
                "frame_id",
            ]
        )

        self.fp.flush()


    def close_output_file(
        self,
    ):
        if self.closed_file:
            return

        self.closed_file = True

        if self.fp is not None:
            self.fp.flush()
            self.fp.close()

            self.fp = None


    # ========================================================
    # ROS Callback
    # ========================================================

    def callback(
        self,
        msg,
    ):
        latitude = float(
            msg.latitude
        )

        longitude = float(
            msg.longitude
        )

        altitude = float(
            msg.altitude
        )

        # ----------------------------------------------------
        # GPS 데이터 유효성 검사
        # ----------------------------------------------------

        if (
            not math.isfinite(
                latitude
            )
            or
            not math.isfinite(
                longitude
            )
        ):
            if self.node is not None:
                self.node.get_logger().warning(
                    "NaN/Inf GPS 좌표가 들어와 제외했습니다."
                )

            return

        if not math.isfinite(
            altitude
        ):
            altitude = 0.0

        # ----------------------------------------------------
        # GPS -> local XY
        # ----------------------------------------------------

        x, y = self.latlong2xy(
            latitude=latitude,
            longitude=longitude,
            map_offset_x=self.map_offset_x,
            map_offset_y=self.map_offset_y,
        )

        # ----------------------------------------------------
        # 최소 거리 검사
        # ----------------------------------------------------

        if self.last_xy is not None:
            dx = (
                x
                -
                self.last_xy[0]
            )

            dy = (
                y
                -
                self.last_xy[1]
            )

            distance = math.hypot(
                dx,
                dy,
            )

            if distance < self.min_distance:
                return

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        stamp = (
            float(
                msg.header.stamp.sec
            )
            +
            float(
                msg.header.stamp.nanosec
            )
            *
            1.0e-9
        )

        frame_id = str(
            msg.header.frame_id
        )

        # ----------------------------------------------------
        # CSV 저장
        # ----------------------------------------------------

        self.writer.writerow(
            [
                f"{x:.12f}",
                f"{y:.12f}",
                f"{altitude:.6f}",
                f"{stamp:.9f}",
                frame_id,
            ]
        )

        self.fp.flush()

        self.last_xy = (
            x,
            y,
        )

        self.count += 1

        # ----------------------------------------------------
        # 첫 메시지 출력
        # ----------------------------------------------------

        if not self.first_message_received:
            self.first_message_received = True

            if self.node is not None:
                self.node.get_logger().info(
                    (
                        "[%s] 첫 GPS 수신\n"
                        "  Topic    : %s\n"
                        "  Latitude : %.10f\n"
                        "  Longitude: %.10f\n"
                        "  X        : %.6f\n"
                        "  Y        : %.6f\n"
                        "  Frame ID : %s"
                    )
                    %
                    (
                        self.line_name.upper(),
                        self.topic,
                        latitude,
                        longitude,
                        x,
                        y,
                        frame_id,
                    )
                )

        # ----------------------------------------------------
        # 100개 단위 상태 출력
        # ----------------------------------------------------

        if (
            self.count
            %
            100
            ==
            0
            and
            self.node is not None
        ):
            self.node.get_logger().info(
                (
                    "[%s] %d points recorded"
                )
                %
                (
                    self.line_name.upper(),
                    self.count,
                )
            )


    # ========================================================
    # ROS 실행
    # ========================================================

    def run(
        self,
    ):
        try:
            import rclpy

            from sensor_msgs.msg import NavSatFix

            from rclpy.qos import (
                qos_profile_sensor_data,
            )

            from rclpy.executors import (
                ExternalShutdownException,
            )

        except ImportError as exc:
            raise RuntimeError(
                "\nROS 2 Python 환경을 찾지 못했습니다.\n"
                "\n먼저 다음을 실행하십시오.\n"
                "\n"
                "source /opt/ros/jazzy/setup.bash\n"
                "source ~/autocarz_jh/install/setup.bash\n"
            ) from exc

        node_name = (
            f"track_"
            f"{self.line_name}"
            f"_ublox_recorder"
        )

        rclpy.init(
            args=None
        )

        try:
            self.node = (
                rclpy.create_node(
                    node_name
                )
            )

            self.subscription = (
                self.node.create_subscription(
                    NavSatFix,
                    self.topic,
                    self.callback,
                    qos_profile_sensor_data,
                )
            )

            self.open_output_file()

            logger = (
                self.node.get_logger()
            )

            logger.info(
                "============================================"
            )

            logger.info(
                "ROS2 Dual u-blox Track Line Recorder"
            )

            logger.info(
                "Recording line : %s"
                %
                self.line_name.upper()
            )

            logger.info(
                "Topic          : %s"
                %
                self.topic
            )

            logger.info(
                "Map offset X   : %.3f"
                %
                self.map_offset_x
            )

            logger.info(
                "Map offset Y   : %.3f"
                %
                self.map_offset_y
            )

            logger.info(
                "Output         : %s"
                %
                self.output_path
            )

            logger.info(
                "Min distance   : %.3f m"
                %
                self.min_distance
            )

            logger.info(
                "첫 GPS 수신 시 좌표를 출력합니다."
            )

            logger.info(
                "Ctrl+C to stop and save"
            )

            logger.info(
                "============================================"
            )

            # ------------------------------------------------
            # Publisher 존재 여부 확인
            # ------------------------------------------------

            publisher_info = (
                self.node.get_publishers_info_by_topic(
                    self.topic
                )
            )

            if not publisher_info:
                logger.warning(
                    (
                        "현재 %s publisher가 확인되지 않습니다.\n"
                        "GPS driver가 실행 중인지 확인하십시오."
                    )
                    %
                    self.topic
                )

            try:
                rclpy.spin(
                    self.node
                )

            except KeyboardInterrupt:
                pass

            except ExternalShutdownException:
                pass

        finally:
            self.close_output_file()

            if self.node is not None:
                try:
                    self.node.destroy_node()

                except Exception:
                    pass

            if rclpy.ok():
                rclpy.shutdown()

            print()

            print(
                "============================================"
            )

            if self.count > 0:
                print(
                    f"[OK] {self.line_name.upper()} 저장 완료"
                )

                print(
                    f"     기록점 : {self.count}"
                )

                print(
                    f"     Topic  : {self.topic}"
                )

                print(
                    f"     파일   : {self.output_path}"
                )

            else:
                print(
                    f"[FAIL] {self.line_name.upper()} 기록점이 0개입니다."
                )

                print(
                    f"       Topic: {self.topic}"
                )

                print()

                print(
                    "아래 명령으로 GPS 토픽을 확인하십시오."
                )

                print()

                print(
                    f"ros2 topic hz {self.topic}"
                )

                print(
                    f"ros2 topic echo {self.topic} --once"
                )

                print()

                print(
                    "전체 GPS 토픽 확인:"
                )

                print(
                    'ros2 topic list | grep -E "ublox|fix"'
                )

            print(
                "============================================"
            )


# ============================================================
# CLI
# ============================================================

def build_arg_parser(
):
    parser = argparse.ArgumentParser(
        description=(
            "ROS2 Dual u-blox NavSatFix 데이터를 "
            "Track LEFT/MIDDLE/RIGHT Raw CSV로 기록"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------
    # record
    # --------------------------------------------------------

    record_parser = (
        subparsers.add_parser(
            "record",
            help="GPS Track Line 기록",
        )
    )

    record_parser.add_argument(
        "--line",
        required=True,
        choices=[
            "left",
            "middle",
            "middle1",
            "middle2",
            "right",
        ],
        help=(
            "3-Line: left / middle / right, "
            "4-Line: left / middle1 / middle2 / right"
        ),
    )

    record_parser.add_argument(
        "--topic",
        default="/ublox1/fix",
        help=(
            "NavSatFix 토픽 "
            "(기본: /ublox1/fix)"
        ),
    )

    record_parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=(
            "CSV 저장 폴더 "
            "(기본: autocarz/path/track_data)"
        ),
    )

    record_parser.add_argument(
        "--map-offset-x",
        type=float,
        default=-125320.0,
        help=(
            "GPS -> XY map offset X "
            "(기본: -125320.0)"
        ),
    )

    record_parser.add_argument(
        "--map-offset-y",
        type=float,
        default=136746.0,
        help=(
            "GPS -> XY map offset Y "
            "(기본: 136746.0)"
        ),
    )

    record_parser.add_argument(
        "--min-distance",
        type=float,
        default=0.05,
        help=(
            "연속 기록점 최소 거리 [m] "
            "(기본: 0.05)"
        ),
    )

    return parser


# ============================================================
# Main
# ============================================================

def main(
):
    parser = (
        build_arg_parser()
    )

    args = (
        parser.parse_args()
    )

    try:
        if args.command == "record":
            output_path = os.path.join(
                os.path.abspath(
                    args.data_dir
                ),
                f"{args.line}_raw.csv",
            )

            recorder = (
                UbloxTrackRecorder(
                    line_name=args.line,
                    output_path=output_path,
                    topic=args.topic,
                    min_distance=args.min_distance,
                    map_offset_x=args.map_offset_x,
                    map_offset_y=args.map_offset_y,
                )
            )

            recorder.run()

    except KeyboardInterrupt:
        print(
            "\n[CANCEL] 사용자에 의해 중단되었습니다."
        )

        sys.exit(
            130
        )

    except Exception as exc:
        print(
            "\n[FAIL] 프로그램 실행"
        )

        print(
            f"       원인: {exc}"
        )

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()