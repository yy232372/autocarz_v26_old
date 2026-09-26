#!/usr/bin/env python3
"""Record a planner path from the live /odom topic.

The output format intentionally matches the Autocarz path files:

    x y 0 1

The fourth value is kept at 1 because the ROI nodes use it as the path
corridor radius.  Samples are distance-thinned so GPS messages at 10 Hz do
not create hundreds of nearly identical rows while the vehicle is stopped.
"""

import argparse
import math
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class PathRecorder(Node):
    def __init__(self, output_path: Path, topic: str, min_distance: float):
        super().__init__('path_recorder')

        if min_distance < 0.0:
            raise ValueError('--min-distance must be non-negative')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.output = output_path.open('w', encoding='utf-8')
        self.min_distance = min_distance

        # planner_vector publishes /odom as RELIABLE/VOLATILE.  Use a larger
        # local queue than the planner so startup bursts are less likely to be
        # dropped by this recorder.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            Odometry,
            topic,
            self._odom_callback,
            qos,
        )
        self.timer = self.create_timer(1.0, self._report)

        self.received = 0
        self.saved = 0
        self.invalid = 0
        self.last_x = None
        self.last_y = None
        self.last_message_x = None
        self.last_message_y = None
        self.spacings = []
        self._closed = False

        self.get_logger().info(
            f'Recording {topic} -> {output_path} '
            f'(min distance {min_distance:.2f} m)'
        )
        self.get_logger().info('Drive the route; press Ctrl+C when finished.')

    def _odom_callback(self, message: Odometry) -> None:
        self.received += 1
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        self.last_message_x = x
        self.last_message_y = y

        if not (math.isfinite(x) and math.isfinite(y)):
            self.invalid += 1
            self.get_logger().warning('Ignoring non-finite /odom position')
            return

        if self.last_x is None:
            spacing = 0.0
        else:
            spacing = math.hypot(x - self.last_x, y - self.last_y)
            if spacing < self.min_distance:
                return
            self.spacings.append(spacing)

        self.output.write(f'{x:.8f} {y:.8f} 0 1\n')
        self.output.flush()
        self.last_x = x
        self.last_y = y
        self.saved += 1

        print(
            f'[SAVE {self.saved:05d}] '
            f'x={x:.8f} y={y:.8f} spacing={spacing:.3f} m',
            flush=True,
        )

    def _report(self) -> None:
        if self.received == 0:
            print('[WAIT] no /odom message received yet', flush=True)
            return

        if self.last_message_x is None:
            return

        print(
            f'[LIVE] received={self.received} saved={self.saved} '
            f'invalid={self.invalid} '
            f'latest=({self.last_message_x:.3f}, {self.last_message_y:.3f})',
            flush=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.output.flush()
        self.output.close()

        if self.spacings:
            average = sum(self.spacings) / len(self.spacings)
            minimum = min(self.spacings)
            maximum = max(self.spacings)
        else:
            average = minimum = maximum = 0.0

        print('\n[PATH CHECK]', flush=True)
        print(f'file: {self.output_path}', flush=True)
        print(
            f'received={self.received} saved={self.saved} invalid={self.invalid}',
            flush=True,
        )
        print(
            f'spacing: avg={average:.3f} m min={minimum:.3f} m '
            f'max={maximum:.3f} m',
            flush=True,
        )
        print('format: x y 0 1', flush=True)
        if self.saved == 0:
            print('RESULT: FAIL (no points were recorded)', flush=True)
        elif self.saved < 2:
            print('RESULT: WARN (only one point; drive farther)', flush=True)
        elif maximum > 3.0:
            print('RESULT: WARN (a jump over 3 m was recorded)', flush=True)
        elif self.invalid:
            print('RESULT: WARN (invalid /odom samples were ignored)', flush=True)
        else:
            print('RESULT: OK (path samples were recorded)', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Record x/y waypoints from /odom')
    parser.add_argument(
        '--output',
        default='/tmp/new_path.txt',
        help='output path file (default: /tmp/new_path.txt)',
    )
    parser.add_argument(
        '--topic',
        default='/odom',
        help='Odometry topic (default: /odom)',
    )
    parser.add_argument(
        '--min-distance',
        type=float,
        default=0.20,
        help='minimum distance between saved points in metres (default: 0.20)',
    )
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = None
    try:
        node = PathRecorder(Path(args.output).expanduser(), args.topic, args.min_distance)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
