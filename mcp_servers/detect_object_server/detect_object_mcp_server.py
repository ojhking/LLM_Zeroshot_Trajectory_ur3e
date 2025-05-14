import os
import numpy as np
import torch
import pyrealsense2 as rs
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sklearn.decomposition import PCA
from typing import Optional, Dict, Any, List # Type Hinting 추가
from mcp.server.fastmcp import FastMCP, Context  # MCP 관련 import
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import asyncio 
import time
# 현재 스크립트 파일 기준 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)
# +++++++++++++ 오프셋 설정 추가 +++++++++++++
# 각 축(W, L, H)에 대한 크기 보정 오프셋 (단위: 미터)
# 필요에 따라 이 값들을 조정하세요.
WIDTH_OFFSET = -0.01  # Width (너비) 오프셋
LENGTH_OFFSET = -0.01 # Length (길이) 오프셋
HEIGHT_OFFSET = -0.02# Height (높이) 오프셋
X_OFFSET = 0.0
Y_OFFSET = 0.48
# ++++++++++++++++++++++++++++++++++++++++++
class GroundedSAM2Wrapper:
    def __init__(
        self,
        dino_model_id="IDEA-Research/grounding-dino-base",
        sam2_config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path = "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Grounding DINO 로딩
        try:
            print("Loading Grounding DINO model...")
            self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(self.device)
            print("Grounding DINO loaded.")
        except Exception as e:
            print(f"Error loading Grounding DINO: {e}")
            raise

        # 2. SAM2 predictor 로딩
        try:
            print("Loading SAM2 model...")
            sam2_model = build_sam2(sam2_config_path, sam2_ckpt_path, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("SAM2 loaded.")
        except Exception as e:
            print(f"Error loading SAM2: {e}")
            raise

    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        try:
            image_pil = Image.fromarray(image)
            image_np = image
        except Exception as e:
            print(f"Error converting NumPy array to PIL Image: {e}")
            raise TypeError("Input image must be a NumPy array.")

        if not text_prompt.strip().endswith("."):
            text_prompt += "."

        inputs = self.dino_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        target_size = [image_pil.size[::-1]] # [H, W]

        detections = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_size,
        )[0]

        boxes = detections["boxes"]
        scores = detections["scores"]
        phrases = detections["labels"]

        if boxes.size(0) == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device), \
                   torch.empty((0, 4), dtype=torch.float, device=self.device), \
                   [], \
                   torch.empty(0, dtype=torch.float, device=self.device)

        self.sam2_predictor.set_image(image_np)
        boxes_np = boxes.cpu().numpy()

        sam_masks, sam_scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_np,
            multimask_output=False,
        )

        if sam_masks.ndim == 4:
             sam_masks = sam_masks.squeeze(1)

        masks_tensor = torch.from_numpy(sam_masks).to(self.device)

        return masks_tensor, boxes, phrases, scores


class RealSenseCapture:
    def __init__(self, depth_w=1024, depth_h=768, color_w=960, color_h=540, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.is_running = False
        self.depth_scale = None
        self.color_intrinsics = None
        self.depth_w, self.depth_h = depth_w, depth_h
        self.color_w, self.color_h = color_w, color_h
        self.fps = fps

        try:
            self.config.enable_stream(rs.stream.depth, self.depth_w, self.depth_h, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.color_w, self.color_h, rs.format.bgr8, self.fps)
        except RuntimeError as e:
            print(f"Error enabling stream: {e}. Check if resolution/fps is supported.")
            self.pipeline = None
            raise

    def start(self):
        if not self.pipeline: return False
        if not self.is_running:
            try:
                print("Starting RealSense pipeline...")
                profile = self.pipeline.start(self.config)
                self.is_running = True

                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()

                align_to = rs.stream.color
                self.align = rs.align(align_to)

                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                self.color_intrinsics = color_stream.get_intrinsics()
                print(f"RealSense started. Depth Scale: {self.depth_scale:.4f}")

                print("Waiting for frames to stabilize...")
                for _ in range(15): # 안정화 시간 필요
                    self.pipeline.wait_for_frames()
                print("Stabilization complete.")
                return True
            except RuntimeError as e:
                print(f"Failed to start pipeline: {e}")
                self.is_running = False
                return False
        return True

    def capture_aligned_frames(self, timeout_ms=5000):
        if not self.is_running:
            print("Error: Pipeline not running.")
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if not frames:
                print("Error: Failed to receive frames (timeout).")
                return None, None

            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # 필터 적용 (노이즈 제거용, l515 내부 함수 사용용)
            spatial = rs.spatial_filter()
            temporal = rs.temporal_filter()
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)

            if not depth_frame or not color_frame:
                print("Error: Failed to get valid aligned frames.")
                return None, None

            depth_image_raw = np.asanyarray(depth_frame.get_data()) # uint16
            color_image = np.asanyarray(color_frame.get_data()) # BGR

            return color_image, depth_image_raw

        except Exception as e:
            print(f"Error during frame capture: {e}")
            return None, None

    def get_intrinsics(self):
        return self.color_intrinsics

    def get_depth_scale(self):
        return self.depth_scale

    def stop(self):
        if self.is_running:
            print("Stopping RealSense pipeline...")
            self.pipeline.stop()
            self.is_running = False
            print("Pipeline stopped.")


# 감지된 물체의 3d 좌표 및 정보 파악
def calculate_object_properties(
    mask: np.ndarray,
    depth_image_raw: np.ndarray,
    intrinsics: rs.intrinsics,
    depth_scale: float
):
    if not intrinsics or depth_scale is None:
        print("Error: Invalid camera intrinsics or depth scale.")
        return None, None
    if mask.shape != depth_image_raw.shape:
        print(f"Error: Mask shape {mask.shape} and depth shape {depth_image_raw.shape} mismatch.")
        return None, None
    if mask.dtype != bool:
        mask = mask > 0

    object_pixels_yx = np.argwhere(mask)
    if object_pixels_yx.size == 0:
        print("Warning: No pixels found in the mask.")
        return None, None

    depth_image_meters = depth_image_raw.astype(np.float32) * depth_scale
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    point_cloud = []
    try:
        for y, x in object_pixels_yx:
            depth = depth_image_meters[y, x]
            if depth > 0.01 and depth < 10.0: # 유효 깊이 범위
                Z_pt = depth
                X_pt = (x - cx) * Z_pt / fx
                Y_pt = (y - cy) * Z_pt / fy
                point_cloud.append([X_pt, Y_pt, Z_pt])

        if len(point_cloud) < 10:
            print(f"Warning: Insufficient valid points ({len(point_cloud)}) for 3D calculations.")
            return None, None

        point_cloud_np = np.array(point_cloud, dtype=np.float32)
        center_3d = np.mean(point_cloud_np, axis=0).tolist()

        pca = PCA(n_components=3)
        pca.fit(point_cloud_np)
        transformed_points = pca.transform(point_cloud_np)
        min_coords_obb = np.min(transformed_points, axis=0)
        max_coords_obb = np.max(transformed_points, axis=0)
        size_obb = max_coords_obb - min_coords_obb
        obb_dims = sorted(size_obb.tolist(), reverse=True) # [L, W, H]

        return center_3d, obb_dims

    except Exception as e:
        print(f"Error during 3D property calculation: {e}")
        return None, None

# ---- MCP 서버 설정 시작 ----

# Lifespan에서 사용할 컨텍스트 정의
@dataclass
class AppContext:
    model_wrapper: GroundedSAM2Wrapper
    camera_capture: RealSenseCapture

# Lifespan 관리자 정의
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """서버 시작 시 모델과 카메라 로드, 종료 시 카메라 해제"""
    print("Initializing resources for MCP server...")
    # 서버 시작 시 초기화
    model_wrapper = None
    camera_capture = None
    try:
        # 모델 로딩 (시간 소요될 수 있음)
        # 실제 환경에서는 에러 처리 강화 필요
        model_wrapper = GroundedSAM2Wrapper()

        # 카메라 초기화 및 시작
        camera_capture = RealSenseCapture()
        # start() 메서드가 동기 함수이므로 asyncio.to_thread 사용 권장
        # 또는 start()를 async로 만들거나, 별도 스레드에서 실행 고려
        # 여기서는 간단하게 직접 호출 (Event Loop Blocking 가능성 있음)
        if not camera_capture.start():
             raise RuntimeError("Failed to start RealSense camera during lifespan init.")

        print("Resources initialized successfully.")
        # 초기화된 객체들을 컨텍스트로 전달
        yield AppContext(model_wrapper=model_wrapper, camera_capture=camera_capture)

    finally:
        print("Shutting down resources...")
        if camera_capture and camera_capture.is_running: # is_running 확인 추가
            try:
                print("Stopping camera capture in separate thread...")
                await asyncio.to_thread(camera_capture.stop) # <<< 비동기적으로 실행
                print("Camera capture stopped.")
            except Exception as e:
                print(f"Error stopping camera capture: {e}") # 종료 시 오류 로깅
        print("Resources shut down.")


# MCP 서버 인스턴스 생성 및 lifespan 연결
mcp = FastMCP("detect_object_mcp_server", lifespan=app_lifespan)

import logging
# 로그 파일 이름, 레벨, 포맷 설정
log_file_path = os.path.join(base_dir, 'mcp_server.log') # 로그 파일을 스크립트와 같은 디렉토리에 저장
logging.basicConfig(level=logging.INFO, # INFO 레벨 이상의 로그를 기록
                    filename=log_file_path, # 파일 이름 지정
                    filemode='a', # 'a'는 append 모드, 'w'는 write 모드 (실행 시마다 덮어씀)
                    format='%(asctime)s - %(levelname)s - %(message)s')
# ++++++++++++++++++++++++++++++++++++++++++

# ... (GroundedSAM2Wrapper, RealSenseCapture, calculate_object_properties 클래스/함수 정의는 동일) ...
# ... (오프셋 설정, AppContext, app_lifespan, mcp 인스턴스 생성 등은 동일) ...
@mcp.tool()
async def find_object_3d_properties(object_name: str, ctx: Context) -> Dict[str, Any]:
    """
    Finds the 3D position (X, Y, Z) of a specified physical object.
    Input: 'object_name'.
    Output: Returns a dictionary containing:
    - 'object_name': The name of the object found (string).
    - 'position': The object's center X,Y,Z coordinate in meters (float).
    """
    tool_start_time = time.monotonic()
    logging.info(f"Tool function started for '{object_name}'")

    # --- Initialize the NEW result structure ---
    result = {
        "object_name": None,
        "position": None  # Changed from x, y, z to a single 'position' key
    }

    # Assuming AppContext setup is correct
    app_context: AppContext = ctx.request_context.lifespan_context
    wrapper = app_context.model_wrapper
    capture = app_context.camera_capture

    try:
        # --- Frame Capture ---
        capture_start = time.monotonic()
        color_image, depth_image_raw = await asyncio.to_thread(
            capture.capture_aligned_frames
        )
        capture_duration = time.monotonic() - capture_start
        logging.info(f"  - Frame Capture Duration: {capture_duration:.3f} seconds")
        if color_image is None or depth_image_raw is None:
            raise RuntimeError("Failed to capture frame from RealSense.")

        intrinsics = capture.get_intrinsics()
        depth_scale = capture.get_depth_scale()
        print(f"fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, ppx={intrinsics.ppx:.1f}, ppy={intrinsics.ppy:.1f}")
        if not intrinsics or depth_scale is None:
            raise RuntimeError("Failed to get camera parameters.")

        # --- Model Prediction ---
        logging.info(f"Running detection for '{object_name}'...")
        predict_start = time.monotonic()
        masks_tensor, boxes_tensor, phrases_list, scores_tensor = await asyncio.to_thread(
            wrapper.predict, color_image, object_name
        )
        predict_duration = time.monotonic() - predict_start
        logging.info(f"  - Model Prediction Duration: {predict_duration:.3f} seconds")

        if not phrases_list:
            logging.warning(f"Object '{object_name}' not found.")
            result["error"] = f"Object '{object_name}' not found."
        else:
            logging.info(f"Found {len(phrases_list)} potential instance(s).")
            scores_np = scores_tensor.cpu().numpy()
            best_match_idx = np.argmax(scores_np)

            selected_phrase = phrases_list[best_match_idx]
            selected_score = float(scores_np[best_match_idx])
            selected_mask_tensor = masks_tensor[best_match_idx]
            selected_mask_np = selected_mask_tensor.cpu().numpy()

            logging.info(f"Selected object: '{selected_phrase}' with score {selected_score:.2f}")
            result["object_name"] = selected_phrase

            if selected_mask_np.dtype != bool:
                selected_mask_np = selected_mask_np > 0.5

            # --- 3D Calculation ---
            logging.info("Calculating 3D properties...")
            calc_start = time.monotonic()
            # calculate_object_properties should return center [Orig Y, Orig X, Orig Z] and dimensions [L,W,H]
            center_3d_coords, obb_dims_LWH = await asyncio.to_thread(
                calculate_object_properties,
                selected_mask_np,
                depth_image_raw,
                intrinsics,
                depth_scale
            )
            calc_duration = time.monotonic() - calc_start
            logging.info(f"  - 3D Calculation Duration: {calc_duration:.3f} seconds")

            # --- Process Results (Modified for combined position) ---
            calculated_x = None
            calculated_y = None
            calculated_z = None
            calculation_error_msg = ""

            if center_3d_coords:
                # Apply offsets and rounding for X and Y
                # NOTE: Swapping index 1 for X and 0 for Y as per original code's logic
                calculated_x = round(float(center_3d_coords[1]), 3) + X_OFFSET
                calculated_y = round(float(center_3d_coords[0]), 3) + Y_OFFSET
            else:
                logging.warning("3D Center coordinate calculation failed.")
                calculation_error_msg += " 3D Center calculation failed."

            if obb_dims_LWH:
                # Extract Height (assuming order L, W, H)
                H = obb_dims_LWH[2]
                # Apply offset only to Height
                adjusted_H = H + HEIGHT_OFFSET # Ensure HEIGHT_OFFSET is defined
                calculated_z = round(float(adjusted_H), 3)
            else:
                 # Adjust error message if height calculation failed
                logging.warning("3D Height (Z) calculation failed.")
                calculation_error_msg += " 3D Height calculation failed."

            # Combine into the 'position' list if all components are valid
            if calculated_x is not None and calculated_y is not None and calculated_z is not None:
                result["position"] = [
                                        round(calculated_x, 3),
                                        round(calculated_y, 3),
                                        round(calculated_z, 3)
                                    ]
                logging.info(f"Calculated final 3D position [X,Y,Z]: {result['position']}")
            else:

                logging.warning("Could not determine full 3D position. Error")


    except Exception as e:
        logging.error(f"Error processing request for '{object_name}': {e}", exc_info=True)

    finally:
        tool_duration = time.monotonic() - tool_start_time
        logging.info(f"Tool function finished for '{object_name}'. Total Duration: {tool_duration:.3f} seconds")

    # --- Return the modified result dictionary ---
    return result

# ---- 서버 실행 ----
if __name__ == "__main__":
    print("Starting MCP server...")
    # 필요한 경우 mcp.run()에 host, port 등 인자 전달 가능
    mcp.run()

