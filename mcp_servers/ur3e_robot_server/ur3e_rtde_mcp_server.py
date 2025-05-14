# ur3e_server.py
import sys
import ast
from typing import List, Union
from mcp.server.fastmcp import FastMCP
# UR3e_RTDE.py 파일에 URController 클래스가 있다고 가정
from UR3e_RTDE import URController

print("UR3e MCP 서버 시작 중...")

# --- FastMCP 초기화 ---
# 서버에 고유한 이름 부여
mcp = FastMCP("ur3e_rtde_mcp_server")

# --- URController 초기화 ---
# 서버 시작 시 로봇 연결 시도
try:
    # 필요하다면 로봇 IP 주소를 설정 가능하게 만들 수 있습니다.
    controller = URController()
    print("URController 초기화 성공.")
    # 선택 사항: 로봇을 알려진 시작 위치로 이동
    # print("로봇을 HOME 위치로 이동 중...")
    # controller.move_home()
    # print("로봇이 HOME 위치에 있습니다.")
except Exception as e:
    print(f"치명적 오류: UR 로봇 연결 실패: {e}")
    print("로봇 전원이 켜져 있고 네트워크에 연결되어 있는지,")
    print("URController (또는 이 스크립트 내)의 IP 주소가 올바른지 확인하세요.")
    print("서버를 종료합니다.")
    sys.exit(1) # 컨트롤러 초기화 실패 시 종료

# --- MCP 도구 정의 ---

@mcp.tool()
def robot_execute_trajectorys(a_list: List[List[Union[int, float]]]) -> str: # <--- Changed signature
    """Moves the robot arm's end-effector along a trajectory path specified by a list of waypoints.
    The robot uses a fixed orientation internally.

    Args:
        a_list (List[List[Union[int, float]]]): A list of waypoints provided directly as a Python list.
                                                 Each waypoint MUST be a list containing exactly three
                                                 numerical values [X, Y, Z] representing
                                                 the target position in METERS.

    Example structure expected for 'a_list' argument in the tool call:
    [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]

    CRITICAL: The input MUST be a list object. Each inner list MUST contain exactly
              three numerical values [X, Y, Z]. The LLM should provide the argument
              directly as a list, not a string representation.
    Returns:
        str: A message indicating success or failure of the trajectory execution.
    """
    global controller
    # String parsing (ast.literal_eval) is removed as input is now a list.
    # Validation is still performed on the list structure and types.
    processed_waypoints: List[List[float]] = []
    gripper_offset: float = 0.08  # 예: 5cm 오프셋

    try:
        # 1. Validate input type (basic check)
        if not isinstance(a_list, list):
            # This might be caught by type hints/Pydantic earlier, but good for clarity
            raise TypeError("Input must be a list of waypoints.")

        # 2. Process and validate each waypoint in the input list
        for i, wp in enumerate(a_list):
            if not isinstance(wp, list):
                raise TypeError(f"Waypoint {i} ('{wp}') is not a list.")
            if len(wp) != 3:
                raise ValueError(f"Waypoint {i} ('{wp}') has an invalid number of elements ({len(wp)}). Expected exactly 3 [X, Y, Z] elements.")
            # Ensure all elements are numbers (int or float)
            if not all(isinstance(n, (int, float)) for n in wp):
                raise ValueError(f"Waypoint {i} ('{wp}') contains non-numeric values. Expected [X, Y, Z] numbers.")

            # Convert all to float for consistency before appending
            processed_waypoints.append([float(n) for n in wp])

        # Handle empty trajectory list
        if not processed_waypoints:
            raise ValueError("Processed trajectory list cannot be empty.")

        # --- 3. Execute the trajectory using URController ---
        print(f"Executing trajectory with {len(processed_waypoints)} waypoints...")

        for i, point_xyz in enumerate(processed_waypoints):
            # Controller's cartesian_move uses fixed rotation internally.
            # Pass dummy rotation [0.0, 0.0, 0.0] as the method might expect 6 elements.
            target_pose_for_controller = [point_xyz[0], point_xyz[1], point_xyz[2], 0.0, 0.0, 0.0]
            print(f"  Moving to waypoint {i+1} (Controller Input): {target_pose_for_controller}")
            try:
                # Call the controller method which applies fixed rotation internally
                controller.cartesian_move(target_pose_for_controller)
                # time.sleep(0.1) # Optional delay
            except Exception as move_e:
                error_msg = f"Error during move to waypoint {i+1} {point_xyz}: {move_e}"
                print(error_msg)
                return f"Failed: {error_msg}" # Return error to LLM

        success_msg = f"Successfully executed trajectory with {len(processed_waypoints)} waypoints."
        print(success_msg)
        return success_msg # Return success to LLM

    except (TypeError, ValueError) as e:
        # Handle validation errors on the input list
        error_msg = f"Input list validation failed. Error: {e}"
        print(error_msg)
        # Consider logging the problematic input 'a_list' here if needed for debugging
        return f"Failed: {error_msg}"
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"An unexpected error occurred in execute_trajectorys: {e}"
        print(error_msg)
        return f"Failed: {error_msg}"

# --- 새로 추가할 TCP 포즈 조회 도구 ---
@mcp.tool()
def robot_get_current_tcp_pose_meters() -> List[float]:
    """Gets the current robot Tool Center Point (TCP) position.

    Returns:
        List[float]: A list containing [X, Y, Z] representing the current TCP
                     position coordinates in METERS. Orientation (Roll, Pitch, Yaw)
                     is not included in the return value.
    """
    global controller # Access the global controller instance
    try:
        # Get the full pose from the controller (returns mm and degrees)
        # Example: [100.0, 200.0, 300.0, 180.0, 0.0, 0.0]
        pose_mm_deg = controller.get_tcp_pose()

        # Validate the received pose format (still expects 6 values from controller)
        if len(pose_mm_deg) != 6:
             raise ValueError(f"Received unexpected pose format from controller: {pose_mm_deg}")

        # Extract only the position part (first 3 elements)
        position_mm = pose_mm_deg[:3]

        # Convert position from millimeters to meters
        position_m = [p / 1000.0 for p in position_mm]
        position_m = [round(p, 3) for p in position_m]

        print(f"Current TCP Position (Meters): {position_m}")
        # Return only the position list [X, Y, Z] in meters
        return position_m

    except Exception as e:
        error_msg = f"Error getting TCP pose: {e}"
        print(error_msg)
        # Re-raise the exception so MCP or the agent can handle it
        raise RuntimeError(error_msg) from e


# --- TODO: 그리퍼 도구 추가 (필요시) ---
# 먼저 URController 클래스 내에 open_gripper(), close_gripper() 메서드를 구현해야 합니다.
# (rtde_c.setStandardDigitalOut() 등 사용)

@mcp.tool()
def robot_turn_on_suction_gripper() -> str:
    """Activates the suction gripper to pick up an object.
    Turns on the vacuum suction.

    Returns:
        str: A message indicating whether the operation was successful ('Suction activated successfully.')
             or failed ('Failed: Error activating suction: ...'), including an error message on failure.
    """
    global controller
    try:
        # controller에 open_gripper 메서드가 있다고 가정
        controller.gripper_on()
        print("실행됨: open_gripper")
        # --- !!! 실제 그리퍼 제어 로직으로 교체 필요 (URController 내 구현) !!! ---
        print("플레이스홀더: 그리퍼 열기 명령 전송됨 (URController에 구현 필요)")
        return "Suction activated successfully."
    except Exception as e:
        error_msg = f"그리퍼 열기 오류: {e}"
        print(error_msg)
        return f"Failed: Error activating suction: {error_msg}"

@mcp.tool()
def robot_turn_off_suction_gripper() -> str:
    """Deactivates the suction gripper to release an object.
    Turns off the vacuum suction.

    Returns:
        str: A message indicating whether the operation was successful ('Suction deactivated successfully.')
             or failed ('Failed: Error deactivating suction: ...'), including an error message on failure.
    """
    global controller
    try:
        # controller에 close_gripper 메서드가 있다고 가정
        controller.gripper_off()
        print("실행됨: close_gripper")
        # --- !!! 실제 그리퍼 제어 로직으로 교체 필요 (URController 내 구현) !!! ---
        print("플레이스홀더: 그리퍼 닫기 명령 전송됨 (URController에 구현 필요)")
        return "Suction deactivated successfully."
    except Exception as e:
        error_msg = f"그리퍼 닫기 오류: {e}"
        print(error_msg)
        return f"Failed: Error deactivating suction: {error_msg}"

@mcp.tool()
def robot_check_attached_gripper() -> str:
    """Checks if an object is currently held securely by the suction gripper.
    Typically involves checking a vacuum sensor reading.

    Returns:
        str: A message indicating if an object is attached ("Object is attached by suction."),
             if no object is attached ("Object is not attached by suction."),
             or if the check failed ("Failed: Gripper check error: ...").
    """
    global controller
    try:
        # controller에 close_gripper 메서드가 있다고 가정
        state = controller.get_gripper_attached_state()
        print("실행됨: check_attached_gripper")
        if state:
            return "Object is attached by suction.r"
        else:
            return "Object is not attached by suction."

            
    except Exception as e:
        error_msg = f"Gripper check error: {e}"
        print(error_msg)
        return f"Failed: {error_msg}"

# --- MCP 서버 시작 ---
if __name__ == "__main__":
    print("MCP 서버 루프 시작...")
    try:
        # 이 함수는 클라이언트 연결/요청을 처리하며 블록됩니다.
        mcp.run()
    except KeyboardInterrupt:
        print("\n사용자에 의해 서버 중지됨 (Ctrl+C).")
    except Exception as e:
        print(f"\nMCP 서버 오류 발생: {e}")
    finally:
        # 서버가 중지될 때 로봇 연결이 닫히도록 보장
        if 'controller' in globals() and controller:
            print("로봇 연결 닫는 중...")
            controller.close()
        print("MCP 서버 종료됨.")