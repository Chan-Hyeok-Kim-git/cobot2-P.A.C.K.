#!/usr/bin/env python3
"""
DRCF 에뮬레이터 모드에서 RViz 그리퍼 시각화를 담당하는 가상 노드.

실제 OnRobot 드라이버의 /onrobot/sendCommand 서비스 인터페이스를 동일하게 구현해
grip_test.py가 모드에 무관하게 같은 서비스를 호출.
"""
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from onrobot_rg_msgs.srv import SetCommand

GRIPPER_OPEN   = -0.558505  # rg2_finger_joint open  (URDF lower limit)
GRIPPER_CLOSED =  0.785398  # rg2_finger_joint closed (URDF upper limit)
PUBLISH_RATE   = 1.0 / 50.0  # 50 Hz — matches real OnRobot driver
SPEED          = 1.0          # rad/s animation speed
DONE_TOL       = 0.01         # rad — "reached target" tolerance


class GripperVirtualNode(Node):

    def __init__(self):
        super().__init__('gripper_virtual_node')
        self._cb_group = ReentrantCallbackGroup()

        self._pub = self.create_publisher(JointState, '/gripper_joint_states', 10)
        self._position = 0.0
        self._target   = 0.0
        self._lock     = threading.Lock()

        self.create_timer(PUBLISH_RATE, self._publish_cb, callback_group=self._cb_group)
        self.create_service(
            SetCommand, '/onrobot/sendCommand',
            self._send_command_cb, callback_group=self._cb_group,
        )
        self.get_logger().info('GripperVirtualNode ready — /onrobot/sendCommand')

    def _send_command_cb(self, req, res):
        if req.command == 'c':
            target = GRIPPER_CLOSED
        elif req.command == 'o':
            target = GRIPPER_OPEN
        else:
            # Numeric fallback: command is a float string (rad). Clamped to URDF
            # limits. Lets demo callers do partial close (block-width grip).
            try:
                target = float(req.command)
            except ValueError:
                res.success = False
                res.message = f'Unknown command: {req.command!r}'
                return res
            target = max(GRIPPER_OPEN, min(GRIPPER_CLOSED, target))

        with self._lock:
            self._target = target

        # 애니메이션이 목표 위치에 도달할 때까지 블로킹
        while True:
            with self._lock:
                done = abs(self._position - target) < DONE_TOL
            if done:
                break
            time.sleep(PUBLISH_RATE)

        res.success = True
        res.message = ''
        return res

    def _publish_cb(self):
        with self._lock:
            diff = self._target - self._position
            step = SPEED * PUBLISH_RATE
            if abs(diff) <= step:
                self._position = self._target
            else:
                self._position += step * (1.0 if diff > 0 else -1.0)
            pos = self._position

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['rg2_finger_joint']
        msg.position = [pos]
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GripperVirtualNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        # launch 종료(SIGINT) 시 rclpy 기본 시그널 핸들러가 이미 context를
        # shutdown한 뒤 ExternalShutdownException이 올라온다 → 중복 호출 시
        # RCLError('rcl_shutdown already called'). ok() 가드로 한 번만 호출.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
