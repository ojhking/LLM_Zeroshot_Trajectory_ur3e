import rtde_control
import rtde_receive
import time
import numpy as np
import math
import rtde_io
from rtde_receive import RTDEReceiveInterface as RTDEReceive


class URController:
    # Class constants
    DEFAULT_IP = "192.168.1.101"
    DEFAULT_VELOCITY = 0.05
    DEFAULT_ACCELERATION = 0.05
    
    # Home positions (degrees)
    ZERO_POSITION = [0, -90, 0, -90, 0, 0]
    HOME_POSITION = [0, -90, -90, -90, 90, 0]

    def __init__(self, robot_ip=DEFAULT_IP):
        """Initialize UR robot controller with RTDE interface"""
        print(f"Connecting to robot({robot_ip})...")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.rtde_io_ = rtde_io.RTDEIOInterface(robot_ip)
        print("Connection successful!")
        
        self.velocity = self.DEFAULT_VELOCITY
        self.acceleration = self.DEFAULT_ACCELERATION

    def move_zero(self):
        """Move to zero position"""
        self.move_joints_degrees(self.ZERO_POSITION)

    def move_home(self):
        """Move to home position"""
        self.move_joints_degrees(self.HOME_POSITION)
        
    def move_joints_degrees(self, joint_positions_degrees):
        """Move joints to specified positions in degrees"""
        radians = np.radians(joint_positions_degrees)
        self.rtde_c.moveJ(radians, self.velocity+1, self.acceleration+1)

    def move_joints_degrees_offset(self, joint_offsets_degrees):
        """Move joints by specified offsets in degrees
        
        Args:
            joint_offsets_degrees (list): 각 관절의 오프셋 값 (degrees)
        """
        current_joints = self.get_current_joints_degrees()
        target_joints = [current + offset for current, offset in zip(current_joints, joint_offsets_degrees)]
        self.move_joints_degrees(target_joints)

    def cartesian_move(self, target_pose: list):
        """
        Move TCP to specified cartesian position
        
        Args:
            target_pose: [x, y, z, roll, pitch, yaw] in mm and degrees
        """
        # Convert mm to meters for position
        # target_pose_m = target_pose[:3]  # x, y, z
        # target_pose_m = [coord / 1000 for coord in target_pose_m]
        
        # Convert euler angles to rotation vector
        # rotation_vector = self.euler_to_rotation_vector(*target_pose[3:])  # roll, pitch, yaw
        # target_pose_m.extend(rotation_vector)
        
        # self.rtde_c.moveL(target_pose, self.velocity, self.acceleration)
        target_pose = target_pose[:3] + [math.pi, 0, 0]
        self.rtde_c.moveJ_IK(target_pose, speed = 0.55, acceleration = 0.5, asynchronous = False)

    def cartesian_move_offset(self, offsets: list):
        """
        Move TCP by specified offsets in cartesian space
        
        Args:
            offsets: [x_offset, y_offset, z_offset, roll_offset, pitch_offset, yaw_offset] in mm and degrees
        """
        current_pose = self.get_tcp_pose()  # 현재 위치 얻기
        target_pose = [current + offset for current, offset in zip(current_pose, offsets)]
        self.cartesian_move(target_pose)  # 새로운 위치로 이동

    def euler_to_rotation_vector(self, roll, pitch, yaw):
        """오일러 각도(degree)를 UR 로봇이 사용하는 회전 벡터(radian)로 변환
        
        Args:
            roll (float): X축 기준 회전 각도 (degrees)
            pitch (float): Y축 기준 회전 각도 (degrees)
            yaw (float): Z축 기준 회전 각도 (degrees)
        
        Returns:
            list: [rx, ry, rz] 회전 벡터 (radians)
        """
        # 1. 입력받은 오일러 각도(degree)를 라디안으로 변환
        rx, ry, rz = np.radians([roll, pitch, yaw])
        
        # 2. 각 축에 대한 기본 회전 행렬 생성
        # X축 회전 행렬
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(rx), -np.sin(rx)],
                      [0, np.sin(rx), np.cos(rx)]])
        
        # Y축 회전 행렬
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]])
        
        # Z축 회전 행렬
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                      [np.sin(rz), np.cos(rz), 0],
                      [0, 0, 1]])
        
        # 3. ZYX 순서로 회전 행렬 결합 (행렬 곱)
        R = Rz @ Ry @ Rx
        
        # 4. 회전 행렬을 회전 벡터로 변환 (Rodrigues' formula)
        # 회전 각도 계산
        theta = np.arccos((np.trace(R) - 1) / 2)
        
        # 회전이 없는 경우
        if theta == 0:
            return [0, 0, 0]
        
        # 회전 축 계산
        rx = (R[2,1] - R[1,2]) / (2 * np.sin(theta))
        ry = (R[0,2] - R[2,0]) / (2 * np.sin(theta))
        rz = (R[1,0] - R[0,1]) / (2 * np.sin(theta))
        
        # 5. 회전 벡터 반환 (축 * 각도)
        return [rx * theta, ry * theta, rz * theta]

    def rotation_vector_to_euler(self, rx, ry, rz):
        """UR 로봇의 회전 벡터(radian)를 오일러 각도(degree)로 변환
        
        Args:
            rx (float): 회전 벡터의 x 성분 (radians)
            ry (float): 회전 벡터의 y 성분 (radians)
            rz (float): 회전 벡터의 z 성분 (radians)
        
        Returns:
            list: [roll, pitch, yaw] 오일러 각도 (degrees)
        """
        # 1. 회전 벡터의 크기(회전 각도) 계산
        theta = np.sqrt(rx*rx + ry*ry + rz*rz)
        
        # 회전이 없는 경우
        if theta < 1e-6:
            return [0, 0, 0]
        
        # 2. 회전 벡터를 회전 행렬로 변환 (Rodrigues' formula)
        # 단위 회전 축 계산
        k = np.array([rx/theta, ry/theta, rz/theta])
        
        # 비대칭 행렬 K 생성
        K = np.array([[0, -k[2], k[1]],
                     [k[2], 0, -k[0]],
                     [-k[1], k[0], 0]])
        
        # Rodrigues' formula를 사용하여 회전 행렬 계산
        # R = I + sin(θ)K + (1-cos(θ))K²
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        
        # 3. 회전 행렬을 오일러 각도로 변환 (ZYX 순서)
        # pitch(β) 계산: -π/2 ≤ β ≤ π/2
        pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
        
        # yaw(α) 계산
        yaw = np.arctan2(R[1,0]/np.cos(pitch), R[0,0]/np.cos(pitch))
        
        # roll(γ) 계산
        roll = np.arctan2(R[2,1]/np.cos(pitch), R[2,2]/np.cos(pitch))
        
        # 4. 라디안을 도(degree)로 변환하여 반환
        return np.degrees([roll, pitch, yaw])
    
    def get_current_joints_degrees(self):
        """Get current joint angles in degrees"""
        return np.degrees(self.rtde_r.getActualQ()).tolist()

    def get_tcp_pose(self):
        """Get current TCP pose [x, y, z, roll, pitch, yaw] in millimeters and degrees"""
        tcp_pose = self.rtde_r.getActualTCPPose()
        position = [x * 1000 for x in tcp_pose[:3]]  # Convert meters to millimeters
        euler_angles = self.rotation_vector_to_euler(*tcp_pose[3:])
        
        return [float(x) for x in list(position) + list(euler_angles)]

    def close(self):
        """Disconnect from the robot"""
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        print("\nRobot disconnected")

    def gripper_on(self):
        """turn on gripper."""
        self.rtde_io_.setToolDigitalOut(0, False)
        self.rtde_io_.setToolDigitalOut(1, True)

    def gripper_off(self):
        """turn off gripper"""
        self.rtde_io_.setToolDigitalOut(0, True)
        self.rtde_io_.setToolDigitalOut(1, False)

    def get_gripper_attached_state(self):
        """물건이 그리퍼에 붙었나 확인합니다. True = 붙음, False = 안붙음"""
        state = self.rtde_r.getDigitalInState(17)
        return state

def main():
    try:
        robot = URController()
        # robot.move_zero()
        # print("Moving to zero position.")
        # robot.move_joints_degrees([0, -90, -90, -90, 90, 0]) # Movint to home position
        # print(robot.get_tcp_pose())
        # robot.gripper_on()
        # robot.get_gripper_attached_state()
        # robot.gripper_off()
        # # time.sleep(5)
        # robot.rtde_c.moveL([0.0, 0.3, 0.4, math.pi, 0, 0])
        # robot.rtde_c.moveL([0.0, -0.3, 0.4, math.pi, 0, 0])
        # robot.rtde_c.moveL([0.0, 0.3, 0.4, math.pi, 0, 0])
        # robot.rtde_c.moveL([0.0, -0.3, 0.4, math.pi, 0, 0])
        # robot.rtde_c.moveL([0.0, 0.3, 0.4, math.pi, 0, 0])
        # robot.cartesian_move([-0.01, 0.40323667063465574, 0.4, math.pi, 0, 0])
        # robot.cartesian_move([0.0, -0.3, 0.4, math.pi, 0, 0]) 
        robot.cartesian_move([0.0, 0.3, 0.3, math.pi, 0, 0]) 
        # robot.cartesian_move([0.0, -0.3, 0.4, math.pi, 0, 0])   # Movint to home position
        # time.sleep(5)
        # robot.rtde_c.moveJ_IK([0.0, 0.3, 0.4, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([0.0, -0.3, 0.4, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([0.0, 0.3, 0.4, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([0.3, 0.2, 0.4, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([-0.1, -0.1, 0.4, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([0.1, 0.1, 0.4, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([0.2, 0.2, 0.4, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([0.3, 0.3, 0.3, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([-0.3, 0.3, 0.3, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([0.3, 0.3, 0.3, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([0.3, -0.3, 0.3, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.rtde_c.moveJ_IK([0.3, 0.3, 0.3, math.pi, 0, 0],speed = 0.55, acceleration = 0.5, asynchronous = False)
        # robot.move_joints_degrees([0, -90, -90, -90, 90, 0]) # Movint to home position
        # print("Joint move")
        # time.sleep(2)
        
        # robot.move_joints_degrees_offset([20, 0, 20, 0, 0, 0])  # Joint1,3 offset
        # print("Joint offset")
        # time.sleep(2)
        

        # robot.cartesian_move([492.566802473547, -132.9753827241311, 488.40143373702693, -179.92272345493853, 0.06812247425190561, -89.96495452981424])  # Movint to home position
        # print("Cartesian move")
        # time.sleep(4)
        # robot.cartesian_move_offset([0.0, 0.0, 200.0, 0.0, 0.0, 0.0])  # z offset
        # print("Cartesian offset")
        # time.sleep(2)
        
        # robot.move_home()
        # print("Moving to Home position.")
        
        
        # print(f"\nCurrent Joint [j1,j2,j3,j4,j5,j6] (degree): {robot.get_current_joints_degrees()}")
        print(f"\nTCP pose [x,y,z,roll,pitch,yaw] (mm): {robot.get_tcp_pose()}")

    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        robot.close()

if __name__ == "__main__":
    main()