from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import random
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import base64
import io
import asyncio
import json
import cv2
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
import mediapipe as mp

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Perception spine (player tracking + court homography). Imported lazily/guarded
# so the API still boots in environments where the heavy CV deps (torch/
# ultralytics) are not installed; movement analysis is simply skipped there.
try:
    from perception.court import CourtCalibration
    from perception.pipeline import analyze_movement
    from perception.annotation import (
        extract_candidate_tracks, extract_frames_for_marking, propagate_ball,
    )
    from perception.timeline import analyze_rally_window
    from perception.selftrain import mine_ball_tracks, scan_video_for_arcs
    from perception.trace import trace_ball_video
    from perception.rallies import segment_rallies, extract_rally_clip
    from perception.rally_events import segment_rallies_v2
    from perception.audio_rallies import segment_rallies_audio
    from perception.players import (get_player_detector, compute_player_court_stats,
                                     compute_court_control)
    from perception.shots_analysis import analyze_shot_patterns
    from perception.identity import extract_player_crops
    PERCEPTION_AVAILABLE = True
except Exception as _perception_err:  # pragma: no cover - optional dependency
    CourtCalibration = None
    analyze_movement = None
    extract_candidate_tracks = None
    extract_frames_for_marking = None
    propagate_ball = None
    analyze_rally_window = None
    mine_ball_tracks = None
    scan_video_for_arcs = None
    trace_ball_video = None
    segment_rallies = None
    extract_rally_clip = None
    segment_rallies_v2 = None
    segment_rallies_audio = None
    get_player_detector = None
    compute_player_court_stats = None
    compute_court_control = None
    analyze_shot_patterns = None
    extract_player_crops = None
    PERCEPTION_AVAILABLE = False
    logging.getLogger(__name__).warning(
        f"Perception spine unavailable, movement analysis disabled: {_perception_err}"
    )

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'squashsense')]

# Create upload directory
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Create the main app
app = FastAPI(title="SquashSense AI API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== MODELS =====================

class ShotData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    shot_type: str  # drive, drop, boast, volley, lob, kill, serve
    timestamp: float  # seconds in video
    player: str  # player1 or player2
    success: bool
    court_position: Optional[str] = None  # front, mid, back
    description: Optional[str] = None

class RallyData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rally_number: int
    start_time: float
    end_time: float
    shot_count: int
    winner: str  # player1 or player2
    winning_shot: str
    shots: List[Dict[str, Any]] = []

class PlayerMovementData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    player: str
    positions: List[Dict[str, float]] = []  # [{x, y, time}]
    court_coverage: float  # percentage
    distance_traveled: float  # estimated meters
    average_speed: float

class SwingAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    player: str
    forehand_count: int = 0
    backhand_count: int = 0
    forehand_quality: float = 0.0  # 0-100
    backhand_quality: float = 0.0
    racket_preparation: str = ""
    follow_through: str = ""

# Training Data Models
class ShotCorrection(BaseModel):
    match_id: str
    shot_index: int
    original_shot_type: str
    corrected_shot_type: str
    original_player: str
    corrected_player: str
    timestamp: float
    frame_base64: Optional[str] = None
    pose_data: Optional[Dict[str, Any]] = None
    corrected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verified: bool = False  # For quality control

class TrainingDataStats(BaseModel):
    total_corrections: int
    corrections_by_shot_type: Dict[str, int]
    verified_samples: int
    model_accuracy_estimate: float

class MatchAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    video_filename: str
    upload_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration: float = 0.0  # video duration in seconds
    status: str = "pending"  # pending, processing, completed, failed
    progress: int = 0  # 0-100
    
    # Player names and identification
    player1_name: str = "Player 1"
    player2_name: str = "Player 2"
    player1_description: str = ""  # How to identify player 1 (shirt color, etc.)
    player2_description: str = ""  # How to identify player 2
    player1_frame: Optional[str] = None  # Base64 frame showing player 1
    player2_frame: Optional[str] = None  # Base64 frame showing player 2
    # Appearance colour signature (HSV torso histogram) — locks identity so the
    # two players are never swapped during detection.
    player1_color_sig: Optional[List[float]] = None
    player2_color_sig: Optional[List[float]] = None
    player1_is_me: bool = False
    player2_is_me: bool = False

    # Court calibration: four floor corners in image pixels, used by the
    # perception spine to map players into real court metres. None = uncalibrated.
    court_calibration: Optional[Dict[str, Any]] = None

    # Analysis results
    total_shots: int = 0
    total_rallies: int = 0
    shots: List[Dict[str, Any]] = []
    rallies: List[Dict[str, Any]] = []
    shot_distribution: Dict[str, int] = {}
    player1_stats: Dict[str, Any] = {}
    player2_stats: Dict[str, Any] = {}
    movement_data: List[Dict[str, Any]] = []
    player_metrics: Dict[str, Any] = {}  # measured movement metrics per player
    swing_analysis: List[Dict[str, Any]] = []
    key_insights: List[str] = []
    thumbnail: Optional[str] = None

class MatchAnalysisCreate(BaseModel):
    title: str

class MatchAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    title: str
    video_filename: str
    upload_time: datetime
    duration: float = 0.0
    status: str = "library"
    progress: int = 0
    player1_name: str = "Player 1"
    player2_name: str = "Player 2"
    player1_description: str = ""
    player2_description: str = ""
    player1_frame: Optional[str] = None
    player2_frame: Optional[str] = None
    player1_is_me: bool = False
    player2_is_me: bool = False
    total_shots: int = 0
    total_rallies: int = 0
    shots: List[Dict[str, Any]] = []
    rallies: List[Dict[str, Any]] = []
    shot_distribution: Dict[str, int] = {}
    player1_stats: Dict[str, Any] = {}
    player2_stats: Dict[str, Any] = {}
    movement_data: List[Dict[str, Any]] = []
    player_metrics: Dict[str, Any] = {}
    court_calibration: Optional[Dict[str, Any]] = None
    swing_analysis: List[Dict[str, Any]] = []
    key_insights: List[str] = []
    thumbnail: Optional[str] = None

# ===================== POSE ESTIMATION =====================

def analyze_pose(frame: np.ndarray) -> Dict[str, Any]:
    """Analyze player pose using MediaPipe"""
    try:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return {"detected": False}
            
            landmarks = results.pose_landmarks.landmark
            
            # Extract key points for squash analysis
            pose_data = {
                "detected": True,
                "landmarks": {},
                "swing_analysis": {}
            }
            
            # Key landmarks for squash
            key_points = {
                "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
                "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
                "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
                "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
                "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
                "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
                "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
                "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
                "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
                "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
                "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
                "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
            }
            
            for name, landmark_id in key_points.items():
                lm = landmarks[landmark_id]
                pose_data["landmarks"][name] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
            
            # Analyze swing mechanics
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Determine dominant hand position (racket hand)
            left_arm_raised = left_wrist.y < left_shoulder.y
            right_arm_raised = right_wrist.y < right_shoulder.y
            
            # Estimate swing type based on arm position relative to body center
            body_center_x = (left_shoulder.x + right_shoulder.x) / 2
            
            if right_arm_raised and right_wrist.visibility > 0.5:
                if right_wrist.x > body_center_x:
                    swing_type = "forehand"
                else:
                    swing_type = "backhand"
            elif left_arm_raised and left_wrist.visibility > 0.5:
                if left_wrist.x < body_center_x:
                    swing_type = "forehand"
                else:
                    swing_type = "backhand"
            else:
                swing_type = "unknown"
            
            # Analyze stance
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            hip_width = abs(left_hip.x - right_hip.x)
            
            if hip_width > 0.15:
                stance = "wide"
            elif hip_width > 0.08:
                stance = "normal"
            else:
                stance = "narrow"
            
            # Calculate arm extension (0-1, 1 being fully extended)
            def calculate_extension(shoulder, elbow, wrist):
                # Simple approximation of arm extension
                arm_length = ((shoulder.x - wrist.x)**2 + (shoulder.y - wrist.y)**2)**0.5
                return min(1.0, arm_length / 0.4)  # Normalize
            
            right_extension = calculate_extension(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            )
            
            pose_data["swing_analysis"] = {
                "swing_type": swing_type,
                "stance": stance,
                "arm_extension": round(right_extension, 2),
                "left_arm_raised": left_arm_raised,
                "right_arm_raised": right_arm_raised
            }
            
            return pose_data
            
    except Exception as e:
        logger.error(f"Pose analysis error: {str(e)}")
        return {"detected": False, "error": str(e)}

def draw_pose_on_frame(frame: np.ndarray, pose_results) -> np.ndarray:
    """Draw pose landmarks on frame"""
    annotated = frame.copy()
    if pose_results and pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
    return annotated

# ===================== AI ANALYSIS =====================

async def analyze_frame_with_ai(frame_base64: str, frame_number: int, context: str = "", player_info: dict = None) -> Dict[str, Any]:
    """Analyze a single frame using GPT-5.2 vision"""
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            logger.error("EMERGENT_LLM_KEY not found")
            return {}
        
        # Build player identification context
        player_context = ""
        p1_name = player_info.get('player1_name', 'Player 1') if player_info else 'Player 1'
        p2_name = player_info.get('player2_name', 'Player 2') if player_info else 'Player 2'
        
        if player_info:
            p1_desc = player_info.get('player1_description', '')
            p2_desc = player_info.get('player2_description', '')
            
            if p1_desc or p2_desc:
                player_context = f"""
IMPORTANT - Player Identification:
- {p1_name} (player1): {p1_desc if p1_desc else 'not specified'}
- {p2_name} (player2): {p2_desc if p2_desc else 'not specified'}
Use these descriptions to correctly identify which player is making each shot.
"""
        
        system_message = f"""You are an expert squash match analyst. Analyze the frame from a squash match video.

The user has identified:
- {p1_name} = player1 (reference image provided first)
- {p2_name} = player2 (reference image provided second)

Identify:
1. Shot type being played (drive, drop, boast, volley, lob, kill, serve, or none if between shots)
2. Which player is making the shot - match them to the reference images
3. Player positions on court (front, mid, back for each player)
4. Swing mechanics if visible (forehand/backhand)
5. Rally state (active, point won, between rallies)
{player_context}
Respond ONLY with valid JSON in this exact format:
{{
    "shot_detected": true/false,
    "shot_type": "drive/drop/boast/volley/lob/kill/serve/none",
    "active_player": "player1/player2/none",
    "player1_position": "front/mid/back",
    "player2_position": "front/mid/back",
    "swing_type": "forehand/backhand/none",
    "racket_prep": "good/average/poor/not_visible",
    "rally_state": "active/point_won/between_rallies",
    "confidence": 0.0-1.0,
    "notes": "brief observation"
}}"""
        
        chat = LlmChat(
            api_key=api_key,
            session_id=f"squash-analysis-{uuid.uuid4()}",
            system_message=system_message
        ).with_model("openai", "gpt-5.2")
        
        # Build image contents - include player reference frames if available
        image_contents = []
        
        # Add player reference frames first for context
        p1_frame = player_info.get('player1_frame') if player_info else None
        p2_frame = player_info.get('player2_frame') if player_info else None
        
        if p1_frame:
            image_contents.append(ImageContent(image_base64=p1_frame))
        if p2_frame:
            image_contents.append(ImageContent(image_base64=p2_frame))
        
        # Add the current frame to analyze
        image_contents.append(ImageContent(image_base64=frame_base64))
        
        prompt_text = f"Analyze this squash match frame (frame #{frame_number}). {context}"
        if p1_frame and p2_frame:
            prompt_text = f"The first image is {p1_name} (player1), the second image is {p2_name} (player2). The third image is the frame to analyze (frame #{frame_number}). {context}"
        elif p1_frame or p2_frame:
            prompt_text = f"The first image is a player reference. The second image is the frame to analyze (frame #{frame_number}). {context}"
        
        user_message = UserMessage(
            text=prompt_text,
            file_contents=image_contents
        )
        
        response = await chat.send_message(user_message)
        
        # Parse JSON response
        try:
            # Clean up response - remove markdown code blocks if present
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            logger.warning(f"Could not parse AI response as JSON: {response[:200]}")
            return {"shot_detected": False, "notes": response[:200]}
            
    except Exception as e:
        logger.error(f"AI analysis error: {str(e)}")
        return {}

async def extract_frames(video_path: str, sample_rate: int = 30) -> List[tuple]:
    """Extract frames from video at specified sample rate"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frame_interval = max(1, int(fps / (sample_rate / 30)))  # Adjust for sample rate
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Resize for efficiency
                frame_resized = cv2.resize(frame, (640, 480))
                # Convert to base64
                _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                timestamp = frame_count / fps if fps > 0 else 0
                
                # Analyze pose
                pose_data = analyze_pose(frame_resized)
                
                frames.append((frame_base64, timestamp, frame_count, frame_resized, pose_data))
            
            frame_count += 1
            
        cap.release()
        return frames, duration
        
    except Exception as e:
        logger.error(f"Frame extraction error: {str(e)}")
        return [], 0

async def generate_thumbnail(video_path: str) -> Optional[str]:
    """Generate thumbnail from first frame of video"""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.resize(frame, (320, 240))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Thumbnail generation error: {str(e)}")
    return None

async def extract_player_frames(video_path: str) -> tuple:
    """Extract frames from different parts of video to represent each player"""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get frame from ~5 seconds into video for player 1
        frame1_pos = min(int(fps * 5), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_pos)
        ret1, frame1 = cap.read()
        
        # Get frame from ~15 seconds into video for player 2
        frame2_pos = min(int(fps * 15), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_pos)
        ret2, frame2 = cap.read()
        
        cap.release()
        
        player1_frame = None
        player2_frame = None
        
        if ret1:
            # Crop to show more of the court/player area
            h, w = frame1.shape[:2]
            frame1 = frame1[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
            frame1 = cv2.resize(frame1, (240, 180))
            _, buffer1 = cv2.imencode('.jpg', frame1, [cv2.IMWRITE_JPEG_QUALITY, 80])
            player1_frame = base64.b64encode(buffer1).decode('utf-8')
        
        if ret2:
            h, w = frame2.shape[:2]
            frame2 = frame2[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
            frame2 = cv2.resize(frame2, (240, 180))
            _, buffer2 = cv2.imencode('.jpg', frame2, [cv2.IMWRITE_JPEG_QUALITY, 80])
            player2_frame = base64.b64encode(buffer2).decode('utf-8')
        
        return player1_frame, player2_frame
    except Exception as e:
        logger.error(f"Player frame extraction error: {str(e)}")
        return None, None

async def process_video_analysis(match_id: str, video_path: str):
    """Background task to process video and run AI analysis"""
    try:
        # Get match info for player descriptions
        match_doc = await db.matches.find_one({"id": match_id}, {"_id": 0})
        player_info = {
            "player1_name": match_doc.get("player1_name", "Player 1"),
            "player2_name": match_doc.get("player2_name", "Player 2"),
            "player1_description": match_doc.get("player1_description", ""),
            "player2_description": match_doc.get("player2_description", ""),
            "player1_frame": match_doc.get("player1_frame"),
            "player2_frame": match_doc.get("player2_frame")
        }
        
        # Update status to processing
        await db.matches.update_one(
            {"id": match_id},
            {"$set": {"status": "processing", "progress": 5}}
        )
        
        # Generate thumbnail
        thumbnail = await generate_thumbnail(video_path)
        if thumbnail:
            await db.matches.update_one(
                {"id": match_id},
                {"$set": {"thumbnail": thumbnail}}
            )
        
        # Extract player frames
        player1_frame, player2_frame = await extract_player_frames(video_path)
        if player1_frame or player2_frame:
            await db.matches.update_one(
                {"id": match_id},
                {"$set": {"player1_frame": player1_frame, "player2_frame": player2_frame}}
            )
        
        # Extract frames
        frames, duration = await extract_frames(video_path, sample_rate=30)
        
        await db.matches.update_one(
            {"id": match_id},
            {"$set": {"duration": duration, "progress": 15}}
        )
        
        if not frames:
            await db.matches.update_one(
                {"id": match_id},
                {"$set": {"status": "failed", "key_insights": ["Failed to extract frames from video"]}}
            )
            return
        
        # Analyze frames with AI
        shots = []
        rallies = []
        current_rally = {"shots": [], "start_time": 0, "rally_number": 1}
        shot_distribution = {"drive": 0, "drop": 0, "boast": 0, "volley": 0, "lob": 0, "kill": 0, "serve": 0}
        
        player1_stats = {
            "shots": 0, "winners": 0, "errors": 0,
            "forehand": 0, "backhand": 0,
            "front_court": 0, "mid_court": 0, "back_court": 0
        }
        player2_stats = {
            "shots": 0, "winners": 0, "errors": 0,
            "forehand": 0, "backhand": 0,
            "front_court": 0, "mid_court": 0, "back_court": 0
        }
        
        movement_data = []
        swing_analyses = []
        
        total_frames = len(frames)
        
        # Sample every Nth frame for AI analysis (to manage API costs)
        sample_every = max(1, total_frames // 20)  # Analyze ~20 frames
        
        for idx, frame_data in enumerate(frames):
            frame_base64, timestamp, frame_num, frame_raw, pose_data = frame_data
            
            if idx % sample_every != 0:
                continue
                
            # Update progress
            progress = 15 + int((idx / total_frames) * 70)
            await db.matches.update_one(
                {"id": match_id},
                {"$set": {"progress": progress}}
            )
            
            # Analyze frame with AI
            context = f"Previous shot: {shots[-1]['shot_type'] if shots else 'none'}"
            
            # Add pose context if available
            if pose_data.get("detected"):
                swing_info = pose_data.get("swing_analysis", {})
                context += f". Pose detected: swing_type={swing_info.get('swing_type', 'unknown')}, stance={swing_info.get('stance', 'unknown')}"
            
            analysis = await analyze_frame_with_ai(frame_base64, frame_num, context, player_info)
            
            if analysis.get("shot_detected"):
                shot_type = analysis.get("shot_type", "drive")
                active_player = analysis.get("active_player", "player1")
                
                # Use pose data to enhance swing type detection
                swing_type = analysis.get("swing_type", "forehand")
                if pose_data.get("detected"):
                    pose_swing = pose_data.get("swing_analysis", {}).get("swing_type")
                    if pose_swing and pose_swing != "unknown":
                        swing_type = pose_swing
                
                shot = {
                    "shot_type": shot_type,
                    "timestamp": timestamp,
                    "player": active_player,
                    "success": analysis.get("confidence", 0.5) > 0.3,
                    "court_position": analysis.get(f"{active_player}_position", "mid"),
                    "swing_type": swing_type,
                    "confidence": analysis.get("confidence", 0.5),
                    "notes": analysis.get("notes", ""),
                    "pose_data": pose_data if pose_data.get("detected") else None,
                    "frame_base64": frame_base64[:100] + "..." if frame_base64 else None,  # Store truncated for reference
                    "user_corrected": False
                }
                shots.append(shot)
                
                # Update shot distribution
                if shot_type in shot_distribution:
                    shot_distribution[shot_type] += 1
                
                # Update player stats
                stats = player1_stats if active_player == "player1" else player2_stats
                stats["shots"] += 1
                if analysis.get("swing_type") == "forehand":
                    stats["forehand"] += 1
                elif analysis.get("swing_type") == "backhand":
                    stats["backhand"] += 1
                
                position = shot.get("court_position", "mid")
                if position == "front":
                    stats["front_court"] += 1
                elif position == "mid":
                    stats["mid_court"] += 1
                else:
                    stats["back_court"] += 1
                
                # Track rally
                current_rally["shots"].append(shot)
                
                # Check if rally ended
                if analysis.get("rally_state") == "point_won":
                    current_rally["end_time"] = timestamp
                    current_rally["shot_count"] = len(current_rally["shots"])
                    current_rally["winner"] = active_player
                    current_rally["winning_shot"] = shot_type
                    rallies.append(current_rally.copy())
                    
                    # Update winner stats
                    if active_player == "player1":
                        player1_stats["winners"] += 1
                    else:
                        player2_stats["winners"] += 1
                    
                    # Start new rally
                    current_rally = {
                        "shots": [],
                        "start_time": timestamp,
                        "rally_number": len(rallies) + 1
                    }
            
            # Movement is no longer fabricated here. Real, court-grounded player
            # trajectories come from the perception spine in a single pass below.

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        # ----- Perception pass: real player movement in court metres -----
        # Requires a court calibration (four floor corners). Without it we cannot
        # ground positions in real coordinates, so we leave movement empty rather
        # than fabricate it.
        player_metrics = {}
        calib = match_doc.get("court_calibration")
        if PERCEPTION_AVAILABLE and calib:
            try:
                # Calibration is stored as normalized (0..1) image fractions;
                # convert to native pixel corners using the real frame size.
                cap = cv2.VideoCapture(video_path)
                vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
                vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
                cap.release()
                court_calib = CourtCalibration.from_normalized(calib, vid_w, vid_h)
                perception = await asyncio.to_thread(
                    analyze_movement,
                    video_path,
                    court_calib,
                )
                movement_data = perception.movement_data
                player_metrics = perception.player_metrics
                logger.info(
                    f"Perception: {perception.frames_processed} frames, "
                    f"{len(movement_data)} movement samples"
                )
            except Exception as perc_err:
                logger.error(f"Perception pass failed: {perc_err}")
        elif not calib:
            logger.info("No court calibration set; skipping movement analysis")

        # Generate key insights
        key_insights = []
        
        total_shots = len(shots)
        if total_shots > 0:
            dominant_shot = max(shot_distribution.items(), key=lambda x: x[1])[0]
            key_insights.append(f"Dominant shot type: {dominant_shot.capitalize()} ({shot_distribution[dominant_shot]} shots)")
        
        if rallies:
            avg_rally_length = sum(r["shot_count"] for r in rallies) / len(rallies)
            key_insights.append(f"Average rally length: {avg_rally_length:.1f} shots")
            
            longest_rally = max(rallies, key=lambda r: r["shot_count"])
            key_insights.append(f"Longest rally: {longest_rally['shot_count']} shots")
        
        if player1_stats["shots"] > 0 or player2_stats["shots"] > 0:
            p1_total = player1_stats["shots"]
            p2_total = player2_stats["shots"]
            if p1_total > p2_total:
                key_insights.append(f"Player 1 was more active with {p1_total} shots vs {p2_total}")
            elif p2_total > p1_total:
                key_insights.append(f"Player 2 was more active with {p2_total} shots vs {p1_total}")
        
        # Court coverage: prefer measured perception metrics; fall back to the
        # shot-position proxy only when movement analysis was unavailable.
        for label, stats in (("player1", player1_stats), ("player2", player2_stats)):
            m = player_metrics.get(label)
            if m:
                stats["court_coverage"] = m["court_coverage_pct"]
                stats["distance_m"] = m["distance_m"]
                stats["avg_speed_ms"] = m["avg_speed_ms"]
                stats["avg_speed_active_ms"] = m.get("avg_speed_active_ms")
                stats["pct_time_moving"] = m.get("pct_time_moving")
                stats["t_dominance_pct"] = m["t_dominance_pct"]
                stats["mean_dist_to_t_m"] = m["mean_dist_to_t_m"]
                stats["pct_left_of_t"] = m.get("pct_left_of_t")
                stats["pct_behind_t"] = m.get("pct_behind_t")
            else:
                cov = stats["front_court"] + stats["mid_court"] + stats["back_court"]
                if cov > 0:
                    stats["court_coverage"] = round(
                        (stats["front_court"] + stats["back_court"]) / cov * 100, 1
                    )

        # Movement-based insight: who controlled the T (lower distance = better).
        p1m, p2m = player_metrics.get("player1"), player_metrics.get("player2")
        if p1m and p2m:
            if p1m["mean_dist_to_t_m"] < p2m["mean_dist_to_t_m"]:
                key_insights.append(
                    f"Player 1 controlled the T better (avg {p1m['mean_dist_to_t_m']}m "
                    f"vs {p2m['mean_dist_to_t_m']}m from the T)"
                )
            else:
                key_insights.append(
                    f"Player 2 controlled the T better (avg {p2m['mean_dist_to_t_m']}m "
                    f"vs {p1m['mean_dist_to_t_m']}m from the T)"
                )
            busier = "Player 1" if p1m["distance_m"] > p2m["distance_m"] else "Player 2"
            key_insights.append(
                f"{busier} covered more ground "
                f"({max(p1m['distance_m'], p2m['distance_m'])}m run)"
            )
            # Tactical positioning (Baclig et al.): time spent behind the T and on
            # the backhand (left) side — high values flag a containable opponent.
            for label, pm in (("Player 1", p1m), ("Player 2", p2m)):
                if pm.get("pct_behind_t") is not None and pm["pct_behind_t"] >= 70:
                    key_insights.append(
                        f"{label} played {pm['pct_behind_t']}% of the time behind the T "
                        f"(pinned deep) and {pm['pct_left_of_t']}% on the left side"
                    )
        
        # Swing analysis summary
        swing_analyses = [
            {
                "player": "player1",
                "forehand_count": player1_stats["forehand"],
                "backhand_count": player1_stats["backhand"],
                "forehand_ratio": round(player1_stats["forehand"] / max(1, player1_stats["shots"]) * 100, 1),
                "backhand_ratio": round(player1_stats["backhand"] / max(1, player1_stats["shots"]) * 100, 1)
            },
            {
                "player": "player2",
                "forehand_count": player2_stats["forehand"],
                "backhand_count": player2_stats["backhand"],
                "forehand_ratio": round(player2_stats["forehand"] / max(1, player2_stats["shots"]) * 100, 1),
                "backhand_ratio": round(player2_stats["backhand"] / max(1, player2_stats["shots"]) * 100, 1)
            }
        ]
        
        # Final update
        await db.matches.update_one(
            {"id": match_id},
            {"$set": {
                "status": "completed",
                "progress": 100,
                "total_shots": total_shots,
                "total_rallies": len(rallies),
                "shots": shots,
                "rallies": rallies,
                "shot_distribution": shot_distribution,
                "player1_stats": player1_stats,
                "player2_stats": player2_stats,
                "movement_data": movement_data,
                "player_metrics": player_metrics,
                "swing_analysis": swing_analyses,
                "key_insights": key_insights
            }}
        )
        
        logger.info(f"Analysis completed for match {match_id}")
        
    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}")
        await db.matches.update_one(
            {"id": match_id},
            {"$set": {"status": "failed", "key_insights": [f"Analysis failed: {str(e)}"]}}
        )

# ===================== API ROUTES =====================

@api_router.get("/")
async def root():
    return {"message": "SquashSense AI API", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@api_router.post("/matches/upload", response_model=MatchAnalysisResponse)
async def upload_match(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = "Untitled Match"
):
    """Upload a squash match video for analysis"""
    
    # Validate file type
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_types}")
    
    # Generate unique filename
    file_ext = file.filename.split(".")[-1] if "." in file.filename else "mp4"
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create match record
    match = MatchAnalysis(
        title=title,
        video_filename=unique_filename,
        status="pending"
    )
    
    match_dict = match.model_dump()
    match_dict["upload_time"] = match_dict["upload_time"].isoformat()
    
    # Generate thumbnail immediately for player selection
    thumbnail = await generate_thumbnail(str(file_path))
    if thumbnail:
        match_dict["thumbnail"] = thumbnail
        match.thumbnail = thumbnail
    
    await db.matches.insert_one(match_dict)
    
    # Don't start analysis - wait for player selection via /set-players or /start-analysis
    
    return match

@api_router.get("/matches", response_model=List[MatchAnalysisResponse])
async def get_matches():
    """Get all match analyses"""
    matches = await db.matches.find({}, {"_id": 0}).sort("upload_time", -1).to_list(100)
    
    for match in matches:
        if isinstance(match.get("upload_time"), str):
            match["upload_time"] = datetime.fromisoformat(match["upload_time"])
    
    return matches

@api_router.get("/matches/{match_id}", response_model=MatchAnalysisResponse)
async def get_match(match_id: str):
    """Get a specific match analysis"""
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    if isinstance(match.get("upload_time"), str):
        match["upload_time"] = datetime.fromisoformat(match["upload_time"])
    
    return match

@api_router.delete("/matches/{match_id}")
async def delete_match(match_id: str):
    """Delete a match analysis"""
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Delete video file
    video_path = UPLOAD_DIR / match["video_filename"]
    if video_path.exists():
        video_path.unlink()
    
    await db.matches.delete_one({"id": match_id})
    
    return {"message": "Match deleted successfully"}

class SetPlayersRequest(BaseModel):
    player1_frame: str
    player2_frame: str

@api_router.post("/matches/{match_id}/set-players")
async def set_player_frames(match_id: str, request: SetPlayersRequest, background_tasks: BackgroundTasks):
    """Set player reference frames and start analysis"""
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Update player frames
    await db.matches.update_one(
        {"id": match_id},
        {"$set": {
            "player1_frame": request.player1_frame,
            "player2_frame": request.player2_frame,
            "status": "processing",
            "progress": 5
        }}
    )
    
    # Start background analysis
    video_path = UPLOAD_DIR / match["video_filename"]
    background_tasks.add_task(process_video_analysis, match_id, str(video_path))
    
    return {"message": "Players set, analysis started"}

class CourtCalibrationRequest(BaseModel):
    # All coords are NORMALIZED image fractions (0..1): each is [fx, fy].
    front_left: List[float]
    front_right: List[float]
    back_right: List[float]
    back_left: List[float]
    tin_left: Optional[List[float]] = None
    tin_right: Optional[List[float]] = None


@api_router.get("/matches/{match_id}/frame")
async def get_video_frame(match_id: str, t: float = 5.0):
    """Return a JPEG frame at time t (seconds) for the calibration UI."""
    import cv2
    from fastapi.responses import Response
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(404, "Match not found")
    path = UPLOAD_DIR / match["video_filename"]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(500, "Could not read frame")
    h, w = frame.shape[:2]
    if w > 1280:
        frame = cv2.resize(frame, (1280, int(h * 1280 / w)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(content=bytes(buf), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=60"})


@api_router.get("/matches/{match_id}/calibrate")
async def get_court_calibration(match_id: str):
    """Return stored calibration for a match."""
    doc = await db.matches.find_one({"id": match_id}, {"_id": 0, "court_calibration": 1})
    if not doc or not doc.get("court_calibration"):
        return {"calibrated": False}
    return {"calibrated": True, "calibration": doc["court_calibration"]}


def _court3d_quality(calib_dict, vw, vh):
    """Build Court3D from a normalized calibration dict; return (court3d, info)."""
    if not CourtCalibration:
        return None, {}
    try:
        from perception.court3d import Court3D
        calib = CourtCalibration.from_normalized(calib_dict, vw, vh)
        c3d = Court3D.from_calibration(calib, vw, vh)
        if c3d is None:
            return None, {"calibration_quality": "bad", "reproj_err_px": None}
        return c3d, {"calibration_quality": c3d.calibration_quality(),
                     "reproj_err_px": c3d.reproj_err_px}
    except Exception as e:
        logger.warning(f"Court3D build failed: {e}")
        return None, {"calibration_quality": "bad", "reproj_err_px": None}


@api_router.post("/matches/{match_id}/set-court")
async def set_court_calibration(match_id: str, calib: CourtCalibrationRequest):
    """Store court calibration (4 floor corners + optional tin line), normalized 0..1.
    Also computes the 3D reprojection quality so the UI can warn on a bad calibration."""
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    video_path = str(UPLOAD_DIR / match["video_filename"])
    import cv2
    cap = cv2.VideoCapture(video_path)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    _, quality = _court3d_quality(calib.model_dump(), vw, vh)

    await db.matches.update_one(
        {"id": match_id},
        {"$set": {"court_calibration": calib.model_dump(), "court_calibration_quality": quality}},
    )
    return {"message": "Court calibration saved", **quality}


@api_router.get("/matches/{match_id}/court-overlay")
async def court_overlay(match_id: str, t: float = 35.0):
    """Return a frame with the 3D court lines (tin, out-lines, floor) projected on
    top — a visual check that the calibration is correct."""
    import cv2
    from fastapi.responses import Response
    from perception.court3d import draw_court_overlay
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(404, "Match not found")
    calib_dict = match.get("court_calibration")
    if not calib_dict:
        raise HTTPException(400, "Calibrate the court first")
    video_path = str(UPLOAD_DIR / match["video_filename"])
    cap = cv2.VideoCapture(video_path)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(500, "Could not read frame")
    c3d, _ = _court3d_quality(calib_dict, vw, vh)
    if c3d is None:
        raise HTTPException(400, "Calibration could not be solved in 3D")
    vis = draw_court_overlay(frame, c3d)
    if vis.shape[1] > 1280:
        vis = cv2.resize(vis, (1280, int(vis.shape[0] * 1280 / vis.shape[1])))
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=bytes(buf), media_type="image/jpeg")


# ===================== PLAYER DETECTION =====================

PLAYER_STATE: Dict[str, Any] = {}   # match_id -> {status, ...}


def _run_player_detection_sync(video_path: str, start_s: float, duration_s: float,
                                calibration, chunk: int = 60) -> Dict:
    """Detect both players across a span and return per-frame positions + stats."""
    import cv2
    from perception.players import PlayerFrame

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = int(start_s * fps)
    n_frames = min(int(duration_s * fps), total - start_f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    detector = get_player_detector()
    all_frames: List[PlayerFrame] = []

    read = 0
    fi = start_f
    while read < n_frames:
        batch = []
        for _ in range(min(chunk, n_frames - read)):
            ret, f = cap.read()
            if not ret:
                break
            h, w = f.shape[:2]
            if w > 1280:
                f = cv2.resize(f, (1280, int(h * 1280 / w)))
            batch.append(f)
        if not batch:
            break
        results = detector.detect_frames(batch, fi, fps)
        all_frames.extend(results)
        fi += len(batch)
        read += len(batch)
    cap.release()

    stats = compute_player_court_stats(all_frames, calibration, fps)

    # Thin the per-frame positions to every 5th frame for the API response
    positions = []
    for pf in all_frames[::5]:
        entry = {"t": round(pf.timestamp, 2), "players": []}
        for pb in pf.players:
            entry["players"].append({
                "id": pb.player_id,
                "cx": round(pb.cx, 1), "cy": round(pb.cy, 1),
                "feet_x": round(pb.feet_x, 1), "feet_y": round(pb.feet_y, 1),
                "w": round(pb.width, 1), "h": round(pb.height, 1),
                "conf": round(pb.conf, 2),
            })
        positions.append(entry)

    return {
        "fps": fps, "span_s": round(n_frames / fps, 1),
        "total_frames": len(all_frames),
        "positions": positions,
        "stats": stats,
    }


async def _run_player_detection(match_id: str, video_path: str,
                                 start_s: float, duration_s: float, calibration):
    PLAYER_STATE[match_id] = {"status": "running"}
    try:
        result = await asyncio.to_thread(
            _run_player_detection_sync, video_path, start_s, duration_s, calibration
        )
        result["status"] = "done"
        PLAYER_STATE[match_id] = result
        await db.player_tracks.replace_one(
            {"match_id": match_id},
            {"match_id": match_id, "start_s": start_s, "result": result,
             "created_at": datetime.now(timezone.utc).isoformat()},
            upsert=True,
        )
        logger.info(f"Player detection done for {match_id}: {result['total_frames']} frames")
    except Exception as e:
        logger.error(f"Player detection failed for {match_id}: {e}", exc_info=True)
        PLAYER_STATE[match_id] = {"status": "failed", "error": str(e)}


class PlayerDetectRequest(BaseModel):
    start_s: float = 0.0
    duration_s: float = 60.0


@api_router.post("/analysis/players/{match_id}")
async def detect_players(match_id: str, req: PlayerDetectRequest):
    """Detect and track both players across a video span. Background."""
    if not PERCEPTION_AVAILABLE or get_player_detector is None:
        raise HTTPException(503, "Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(404, "Match not found")
    if PLAYER_STATE.get(match_id, {}).get("status") == "running":
        return {"message": "Already running", "state": PLAYER_STATE[match_id]}

    video_path = str(UPLOAD_DIR / match["video_filename"])
    calib_dict = match.get("court_calibration")
    calibration = None
    if calib_dict and CourtCalibration:
        import cv2
        cap = cv2.VideoCapture(video_path)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        try:
            calibration = CourtCalibration.from_normalized(calib_dict, vw, vh)
        except Exception:
            calibration = None

    asyncio.create_task(
        _run_player_detection(match_id, video_path, req.start_s, req.duration_s, calibration)
    )
    return {"message": "Player detection started", "match_id": match_id}


@api_router.get("/analysis/players/{match_id}")
async def get_player_tracks(match_id: str):
    """Live status or last stored player detection result."""
    state = PLAYER_STATE.get(match_id)
    if state:
        return state
    doc = await db.player_tracks.find_one({"match_id": match_id}, {"_id": 0})
    if doc:
        return {**doc["result"], "status": "done"}
    return {"status": "idle"}


# ===================== COURT CONTROL (tactical movement) =====================

COURT_CONTROL_STATE: Dict[str, Any] = {}   # match_id -> {status, ...}


def _court_control_sync(video_path: str, rally_windows: List[tuple],
                         calibration, ref_sigs=None, max_seconds: float = 240.0) -> Dict:
    """Run player detection over the ACTIVE-PLAY (rally) windows only, then
    compute tactical court-control metrics. Skips the between-point walking so
    the movement stats reflect real play.

    ``ref_sigs`` {1,2 -> colour signature} locks player identity by appearance.
    Spectators are filtered out by the court-bounds predicate (needs calibration).
    """
    import cv2
    from perception.players import PlayerFrame
    from perception.identity import _on_court_filter

    detector = get_player_detector()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc_w = min(native_w, 1280)
    on_court = _on_court_filter(calibration, native_w, native_h, proc_w)

    # Player feet come from PROCESSED (<=1280px) frames, but `calibration` is in
    # NATIVE pixels — scale it into processed space so the homography matches the
    # coordinate space of the detections (else everyone maps off-court → zeros).
    proc_calibration = calibration
    if calibration is not None and native_w:
        from perception.court import CourtCalibration as _CC
        s = proc_w / native_w

        def _sc(pt):
            return (pt[0] * s, pt[1] * s) if pt is not None else None
        proc_calibration = _CC(
            front_left=_sc(calibration.front_left), front_right=_sc(calibration.front_right),
            back_right=_sc(calibration.back_right), back_left=_sc(calibration.back_left),
            tin_left=_sc(calibration.tin_left), tin_right=_sc(calibration.tin_right))

    all_frames: List[PlayerFrame] = []
    processed_s = 0.0

    for (ws, we) in rally_windows:
        if processed_s >= max_seconds:
            break
        dur = we - ws
        if dur <= 0:
            continue
        dur = min(dur, max_seconds - processed_s)
        start_f = int(ws * fps)
        n = int(dur * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        # Read this rally window in sub-chunks to bound memory
        read = 0
        fi = start_f
        while read < n:
            batch = []
            for _ in range(min(60, n - read)):
                ret, f = cap.read()
                if not ret:
                    break
                h, w = f.shape[:2]
                if w > 1280:
                    f = cv2.resize(f, (1280, int(h * 1280 / w)))
                batch.append(f)
            if not batch:
                break
            all_frames.extend(detector.detect_frames(batch, fi, fps, ref_sigs, on_court))
            fi += len(batch)
            read += len(batch)
        processed_s += dur

    cap.release()

    control = compute_court_control(all_frames, proc_calibration, fps)
    control["active_play_s"] = round(processed_s, 1)
    control["rally_count"] = len(rally_windows)
    return control


async def _run_court_control(match_id: str, video_path: str,
                              rally_windows: List[tuple], calibration, ref_sigs=None):
    COURT_CONTROL_STATE[match_id] = {"status": "running"}
    try:
        result = await asyncio.to_thread(
            _court_control_sync, video_path, rally_windows, calibration, ref_sigs
        )
        result["status"] = "done"
        COURT_CONTROL_STATE[match_id] = result
        await db.court_control.replace_one(
            {"match_id": match_id},
            {"match_id": match_id, "result": result,
             "created_at": datetime.now(timezone.utc).isoformat()},
            upsert=True,
        )
        logger.info(f"Court control done for {match_id}: "
                    f"{result.get('active_play_s')}s active play analyzed")
    except Exception as e:
        logger.error(f"Court control failed for {match_id}: {e}", exc_info=True)
        COURT_CONTROL_STATE[match_id] = {"status": "failed", "error": str(e)}


@api_router.post("/analysis/court-control/{match_id}")
async def start_court_control(match_id: str):
    """Analyze player court control over the match's detected rally windows.

    Requires rally segmentation to have run first (uses its active-play windows)
    and a court calibration (for court-metre metrics).
    """
    if not PERCEPTION_AVAILABLE or compute_court_control is None:
        raise HTTPException(503, "Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(404, "Match not found")
    if COURT_CONTROL_STATE.get(match_id, {}).get("status") == "running":
        return {"message": "Already running"}

    # Pull rally windows from the latest segmentation
    seg = await db.rally_segments.find_one({"match_id": match_id}, {"_id": 0})
    rally_windows: List[tuple] = []
    if seg and seg.get("result", {}).get("rallies"):
        rally_windows = [(r["start_t"], r["end_t"]) for r in seg["result"]["rallies"]]
    if not rally_windows:
        raise HTTPException(400, "No rallies found — run rally segmentation first.")

    video_path = str(UPLOAD_DIR / match["video_filename"])
    calibration = await _load_calibration(match, video_path)
    ref_sigs = _load_ref_sigs(match)

    asyncio.create_task(
        _run_court_control(match_id, video_path, rally_windows, calibration, ref_sigs)
    )
    return {"message": "Analyzing court control", "rally_count": len(rally_windows)}


@api_router.get("/analysis/court-control/{match_id}")
async def get_court_control(match_id: str):
    """Live status or last stored court-control result."""
    state = COURT_CONTROL_STATE.get(match_id)
    if state:
        return state
    doc = await db.court_control.find_one({"match_id": match_id}, {"_id": 0})
    if doc:
        return {**doc["result"], "status": "done"}
    return {"status": "idle"}


# ===================== SHOT PATTERNS & ERROR ZONES =====================

SHOT_PATTERN_STATE: Dict[str, Any] = {}


async def _run_shot_patterns(match_id: str, video_path: str, rally_windows,
                              outcomes: Dict[str, str], calibration, setup: str,
                              ref_sigs=None, rally_ids=None):
    SHOT_PATTERN_STATE[match_id] = {"status": "running"}
    try:
        result = await asyncio.to_thread(
            analyze_shot_patterns, video_path, rally_windows, outcomes, calibration,
            setup, ref_sigs, 240.0, rally_ids
        )
        result["status"] = "done"
        SHOT_PATTERN_STATE[match_id] = result
        await db.shot_patterns.replace_one(
            {"match_id": match_id},
            {"match_id": match_id, "result": result,
             "created_at": datetime.now(timezone.utc).isoformat()},
            upsert=True,
        )
        logger.info(f"Shot patterns done for {match_id}")
    except Exception as e:
        logger.error(f"Shot patterns failed for {match_id}: {e}", exc_info=True)
        SHOT_PATTERN_STATE[match_id] = {"status": "failed", "error": str(e)}


@api_router.post("/analysis/shot-patterns/{match_id}")
async def start_shot_patterns(match_id: str):
    """Analyze shot-origin tendencies + error zones over the rally windows."""
    if not PERCEPTION_AVAILABLE or analyze_shot_patterns is None:
        raise HTTPException(503, "Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(404, "Match not found")
    if SHOT_PATTERN_STATE.get(match_id, {}).get("status") == "running":
        return {"message": "Already running"}

    seg = await db.rally_segments.find_one({"match_id": match_id}, {"_id": 0})
    if not seg or not seg.get("result", {}).get("rallies"):
        raise HTTPException(400, "No rallies found — run rally segmentation first.")
    rallies = seg["result"]["rallies"]
    rally_windows = [(r["start_t"], r["end_t"]) for r in rallies]
    rally_ids = [r["rally_id"] for r in rallies]
    outcomes = seg.get("outcomes", {}) or {}

    video_path = str(UPLOAD_DIR / match["video_filename"])
    calibration = await _load_calibration(match, video_path)
    ref_sigs = _load_ref_sigs(match)

    asyncio.create_task(
        _run_shot_patterns(match_id, video_path, rally_windows, outcomes,
                           calibration, _match_setup(match), ref_sigs, rally_ids)
    )
    return {"message": "Analyzing shot patterns", "rally_count": len(rally_windows),
            "tagged_outcomes": len(outcomes)}


@api_router.get("/analysis/shot-patterns/{match_id}")
async def get_shot_patterns(match_id: str):
    """Live status or last stored shot-pattern result."""
    state = SHOT_PATTERN_STATE.get(match_id)
    if state:
        return state
    doc = await db.shot_patterns.find_one({"match_id": match_id}, {"_id": 0})
    if doc:
        return {**doc["result"], "status": "done"}
    return {"status": "idle"}


# ===================== SCOUTING REPORT (LLM reasoning layer) =====================

async def _generate_llm_narrative(prompt: str) -> Optional[str]:
    """Generate the scouting narrative with Claude. Returns None if no key/error."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as e:
        logger.error(f"LLM narrative generation failed: {e}")
        return None


@api_router.post("/analysis/scouting/{match_id}")
async def generate_scouting_report(match_id: str):
    """Generate a coached scouting report from all stored analyses.

    Always returns a deterministic rule-based report; additionally returns an
    LLM-written narrative when an Anthropic key is configured.
    """
    from scouting import build_findings, build_deterministic_report, build_llm_prompt

    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(404, "Match not found")

    seg = await db.rally_segments.find_one({"match_id": match_id}, {"_id": 0})
    rally_result = (seg or {}).get("result")
    outcomes = (seg or {}).get("outcomes", {}) or {}

    cc_doc = await db.court_control.find_one({"match_id": match_id}, {"_id": 0})
    court_control = (cc_doc or {}).get("result")

    sp_doc = await db.shot_patterns.find_one({"match_id": match_id}, {"_id": 0})
    shot_patterns = (sp_doc or {}).get("result")

    if not rally_result:
        raise HTTPException(400, "Run rally segmentation first to build a scouting report.")

    facts = build_findings(match, rally_result, outcomes, court_control, shot_patterns)
    deterministic = build_deterministic_report(facts)

    narrative = await _generate_llm_narrative(build_llm_prompt(facts))

    result = {
        "facts": facts,
        "deterministic": deterministic,
        "narrative": narrative,
        "llm_used": narrative is not None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.scouting_reports.replace_one(
        {"match_id": match_id}, {"match_id": match_id, **result}, upsert=True
    )
    return result


@api_router.get("/analysis/scouting/{match_id}")
async def get_scouting_report(match_id: str):
    """Return the last stored scouting report, if any."""
    doc = await db.scouting_reports.find_one({"match_id": match_id}, {"_id": 0})
    if doc:
        return {**doc, "status": "done"}
    return {"status": "idle"}


# ===================== 3D RALLY TIMELINE (full-stack integration) =====================

TIMELINE3D_STATE: Dict[str, Any] = {}


def _timeline3d_sync(video_path: str, rallies, court3d, setup, ref_sigs, max_rallies=8):
    from perception.timeline3d import build_rally_timeline_3d
    out = []
    for r in rallies[:max_rallies]:
        tl = build_rally_timeline_3d(
            video_path, r["rally_id"], r["start_t"], r["end_t"], court3d, setup, ref_sigs)
        out.append(tl)
    # aggregate a self-assessed quality (mean consistency px across rallies)
    cons = [t.get("mean_consistency_px", 0) for t in out if t.get("mean_consistency_px")]
    return {"rallies": out,
            "mean_consistency_px": round(float(np.mean(cons)), 1) if cons else None,
            "quality_note": ("3D reconstruction is reliable" if cons and np.mean(cons) < 15
                             else "low-confidence — improve calibration + ball model (flywheel)")}


async def _run_timeline3d(match_id, video_path, rallies, court3d, setup, ref_sigs):
    TIMELINE3D_STATE[match_id] = {"status": "running"}
    try:
        res = await asyncio.to_thread(_timeline3d_sync, video_path, rallies, court3d, setup, ref_sigs)
        res["status"] = "done"
        TIMELINE3D_STATE[match_id] = res
        await db.timeline3d.replace_one({"match_id": match_id},
            {"match_id": match_id, "result": res,
             "created_at": datetime.now(timezone.utc).isoformat()}, upsert=True)
        logger.info(f"3D timeline done for {match_id}")
    except Exception as e:
        logger.error(f"3D timeline failed for {match_id}: {e}", exc_info=True)
        TIMELINE3D_STATE[match_id] = {"status": "failed", "error": str(e)}


@api_router.post("/analysis/timeline3d/{match_id}")
async def start_timeline3d(match_id: str):
    """Run the full Layer 1-6 stack on the match's rallies → canonical 3D Rally
    Timelines (ball events, shots, outcomes), each with a confidence. Needs a
    court calibration that solves in 3D."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(503, "Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(404, "Match not found")
    seg = await db.rally_segments.find_one({"match_id": match_id}, {"_id": 0})
    rallies = (seg or {}).get("result", {}).get("rallies", [])
    if not rallies:
        raise HTTPException(400, "Run rally segmentation first.")
    calib_dict = match.get("court_calibration")
    if not calib_dict:
        raise HTTPException(400, "Calibrate the court first (3D timeline needs it).")
    video_path = str(UPLOAD_DIR / match["video_filename"])
    import cv2
    cap = cv2.VideoCapture(video_path)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    c3d, _ = _court3d_quality(calib_dict, vw, vh)
    if c3d is None:
        raise HTTPException(400, "Calibration could not be solved in 3D — recalibrate.")
    ref_sigs = _load_ref_sigs(match)
    asyncio.create_task(_run_timeline3d(match_id, video_path, rallies, c3d,
                                        _match_setup(match), ref_sigs))
    return {"message": "Building 3D rally timelines", "rallies": len(rallies)}


@api_router.get("/analysis/timeline3d/{match_id}")
async def get_timeline3d(match_id: str):
    state = TIMELINE3D_STATE.get(match_id)
    if state:
        return state
    doc = await db.timeline3d.find_one({"match_id": match_id}, {"_id": 0})
    if doc:
        return {**doc["result"], "status": "done"}
    return {"status": "idle"}


# ===================== SCOREBOARD (Layer 6 rules engine, live) =====================

class ScoreboardRequest(BaseModel):
    first_server: int = 1
    target: int = 11
    best_of: int = 5


@api_router.post("/analysis/scoreboard/{match_id}")
async def compute_scoreboard(match_id: str, req: ScoreboardRequest):
    """Run the match's tagged rally outcomes through the deterministic Squash Brain
    (PAR scoring, serve alternation, games, best-of-5) and return the full
    scoreboard + per-rally running score. Manual tags are the source of truth."""
    from squash_brain import outcome_from_manual_tag, score_match

    seg = await db.rally_segments.find_one({"match_id": match_id}, {"_id": 0})
    if not seg:
        raise HTTPException(400, "No rallies — run segmentation and tag outcomes first.")
    rallies = seg.get("result", {}).get("rallies", [])
    outcomes = seg.get("outcomes", {}) or {}

    # Build the ordered outcome list (skip untagged + warmup/let handled by engine)
    ordered = sorted(rallies, key=lambda r: r["rally_id"])
    rally_outcomes = []
    rally_ids = []
    for r in ordered:
        tag = outcomes.get(str(r["rally_id"]))
        if not tag:
            continue
        rally_outcomes.append(outcome_from_manual_tag(tag))
        rally_ids.append(r["rally_id"])

    if not rally_outcomes:
        return {"tagged": 0, "message": "No rally outcomes tagged yet.",
                "final": None, "running": []}

    result = score_match(rally_outcomes, first_server=req.first_server,
                         target=req.target, best_of=req.best_of)
    # attach rally_id to each running entry
    for entry, rid in zip(result["running"], rally_ids):
        entry["rally_id"] = rid
    result["tagged"] = len(rally_outcomes)
    p1n = seg.get("player1_name")  # not stored here; names come from match
    return result


# ===================== PLAYER IDENTITY (name the two players) =====================

IDENTIFY_STATE: Dict[str, Any] = {}


async def _run_identify(match_id: str, video_path: str, calibration):
    IDENTIFY_STATE[match_id] = {"status": "running"}
    try:
        result = await asyncio.to_thread(
            extract_player_crops, video_path, calibration, 20.0, 80.0, 35
        )
        result["status"] = "done"
        IDENTIFY_STATE[match_id] = result
        logger.info(f"Player identification for {match_id}: ok={result.get('ok')}")
    except Exception as e:
        logger.error(f"Player identification failed for {match_id}: {e}", exc_info=True)
        IDENTIFY_STATE[match_id] = {"status": "failed", "ok": False, "error": str(e)}


@api_router.post("/analysis/identify-players/{match_id}")
async def identify_players(match_id: str):
    """Find a clean frame and return a crop + colour signature for each on-court
    player, so the user can name them. Background (player detection takes ~20s)."""
    if not PERCEPTION_AVAILABLE or extract_player_crops is None:
        raise HTTPException(503, "Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(404, "Match not found")
    if IDENTIFY_STATE.get(match_id, {}).get("status") == "running":
        return {"message": "Already running"}
    video_path = str(UPLOAD_DIR / match["video_filename"])
    calibration = await _load_calibration(match, video_path)
    IDENTIFY_STATE[match_id] = {"status": "running"}
    asyncio.create_task(_run_identify(match_id, video_path, calibration))
    return {"message": "Identifying players", "calibrated": calibration is not None}


@api_router.get("/analysis/identify-players/{match_id}")
async def get_identify_players(match_id: str):
    """Status / result of the player-identification scan."""
    return IDENTIFY_STATE.get(match_id, {"status": "idle"})


class PlayerNameEntry(BaseModel):
    slot: int                       # 1 = left-most crop, 2 = right-most
    name: str
    is_me: bool = False
    crop_b64: Optional[str] = None
    color_sig: Optional[List[float]] = None


class SavePlayersRequest(BaseModel):
    players: List[PlayerNameEntry]


@api_router.post("/matches/{match_id}/save-players")
async def save_players(match_id: str, req: SavePlayersRequest):
    """Persist the named players (names, crop images, colour signatures, 'me' tag).

    slot 1 → player1, slot 2 → player2. Colour signatures lock identity during
    detection so the two players never get swapped.
    """
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(404, "Match not found")

    update: Dict[str, Any] = {}
    by_slot = {p.slot: p for p in req.players}
    for slot, field in ((1, "player1"), (2, "player2")):
        p = by_slot.get(slot)
        if not p:
            continue
        update[f"{field}_name"] = p.name.strip() or f"Player {slot}"
        if p.crop_b64:
            update[f"{field}_frame"] = p.crop_b64
        if p.color_sig:
            update[f"{field}_color_sig"] = p.color_sig
        update[f"{field}_is_me"] = p.is_me

    await db.matches.update_one({"id": match_id}, {"$set": update})
    return {"ok": True, "names": {k: v for k, v in update.items() if k.endswith("_name")}}


# ===================== FULL MATCH ANALYSIS (one-click orchestrator) =====================
# Chains: rally segmentation → court control → shot patterns → scouting report.

FULL_ANALYSIS_STATE: Dict[str, Any] = {}

FULL_STAGES = ["rallies", "court_control", "shot_patterns", "scouting"]
FULL_STAGE_LABELS = {
    "rallies": "Segmenting rallies",
    "court_control": "Analyzing court control",
    "shot_patterns": "Analyzing shot patterns",
    "scouting": "Writing scouting report",
}


def _set_full_stage(match_id: str, stage: str, done: List[str]):
    FULL_ANALYSIS_STATE[match_id] = {
        "status": "running",
        "stage": stage,
        "stage_label": FULL_STAGE_LABELS.get(stage, stage),
        "stages_done": list(done),
        "total_stages": len(FULL_STAGES),
    }


async def _run_full_analysis(match_id: str, match: dict, video_path: str,
                              start_s: float, duration_s: float):
    setup = _match_setup(match)
    calibration = await _load_calibration(match, video_path)
    ref_sigs = _load_ref_sigs(match)
    done: List[str] = []
    try:
        # ── Stage 1: rally segmentation (+ clip extraction) ────────────────────
        # Reuse an existing segmentation if one is stored — the user may have
        # merged/split/tagged it by hand. Re-segmenting here would silently
        # discard that curation and orphan the outcome tags (keyed by rally_id).
        # Explicit re-segmentation is a separate, deliberate action
        # (POST /analysis/rallies). Only segment from scratch when none exists.
        _set_full_stage(match_id, "rallies", done)
        prior = await db.rally_segments.find_one({"match_id": match_id}, {"_id": 0})
        prior_rallies = ((prior or {}).get("result") or {}).get("rallies") or []
        outcomes = (prior or {}).get("outcomes", {}) or {}
        if prior_rallies:
            seg_result = prior["result"]
            seg_result["status"] = "done"
            start_s = prior.get("start_s", start_s)
            RALLY_STATE[match_id] = seg_result
        else:
            # AUDIO is the primary signal — the ball-strike "thwack" segments
            # rallies far more reliably than visual ball tracking, which throws
            # false positives in the between-point pauses and fuses real rallies
            # into blobs. Same order as the Rallies-tab segmenter. Fall back to
            # the visual segmenters only when there's no usable audio track.
            seg_result = None
            if segment_rallies_audio is not None:
                audio_res = await asyncio.to_thread(
                    segment_rallies_audio, video_path, start_s, duration_s
                )
                if audio_res.get("rallies"):
                    seg_result = audio_res
            if seg_result is None:
                if segment_rallies_v2 is not None:
                    seg_result = await asyncio.to_thread(
                        segment_rallies_v2, video_path, start_s, duration_s, setup, calibration
                    )
                else:
                    seg_result = await asyncio.to_thread(
                        segment_rallies, video_path, start_s, duration_s, setup
                    )
            seg_result["status"] = "done"
            seg_result["start_s"] = start_s
            RALLY_STATE[match_id] = seg_result
            await db.rally_segments.replace_one(
                {"match_id": match_id},
                {"match_id": match_id, "start_s": start_s, "result": seg_result,
                 "outcomes": outcomes,
                 "created_at": datetime.now(timezone.utc).isoformat()},
                upsert=True,
            )
        rallies = seg_result.get("rallies", [])
        rally_windows = [(r["start_t"], r["end_t"]) for r in rallies]
        # Extract only clips that are missing — never overwrite existing ones.
        clip_dir = RALLY_CLIPS_DIR / match_id
        clip_dir.mkdir(exist_ok=True)
        for r in rallies:
            out = clip_dir / f"rally_{r['rally_id']}.mp4"
            if not out.exists():
                asyncio.create_task(asyncio.to_thread(
                    extract_rally_clip, video_path, r["start_t"], r["end_t"], str(out)))
        done.append("rallies")

        if not rally_windows:
            FULL_ANALYSIS_STATE[match_id] = {
                "status": "done", "stage": "done", "stages_done": done,
                "total_stages": len(FULL_STAGES),
                "note": "No rallies detected — nothing further to analyze.",
            }
            return

        # ── Stage 2: court control ─────────────────────────────────────────────
        _set_full_stage(match_id, "court_control", done)
        cc = await asyncio.to_thread(
            _court_control_sync, video_path, rally_windows, calibration, ref_sigs)
        cc["status"] = "done"
        COURT_CONTROL_STATE[match_id] = cc
        await db.court_control.replace_one(
            {"match_id": match_id},
            {"match_id": match_id, "result": cc,
             "created_at": datetime.now(timezone.utc).isoformat()}, upsert=True)
        done.append("court_control")

        # ── Stage 3: shot patterns ─────────────────────────────────────────────
        _set_full_stage(match_id, "shot_patterns", done)
        rally_ids = [r["rally_id"] for r in rallies]
        sp = await asyncio.to_thread(
            analyze_shot_patterns, video_path, rally_windows, outcomes, calibration,
            setup, ref_sigs, 240.0, rally_ids)
        sp["status"] = "done"
        SHOT_PATTERN_STATE[match_id] = sp
        await db.shot_patterns.replace_one(
            {"match_id": match_id},
            {"match_id": match_id, "result": sp,
             "created_at": datetime.now(timezone.utc).isoformat()}, upsert=True)
        done.append("shot_patterns")

        # ── Stage 4: scouting report ───────────────────────────────────────────
        _set_full_stage(match_id, "scouting", done)
        from scouting import build_findings, build_deterministic_report, build_llm_prompt
        facts = build_findings(match, seg_result, outcomes, cc, sp)
        deterministic = build_deterministic_report(facts)
        narrative = await _generate_llm_narrative(build_llm_prompt(facts))
        scouting_doc = {
            "facts": facts, "deterministic": deterministic, "narrative": narrative,
            "llm_used": narrative is not None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        await db.scouting_reports.replace_one(
            {"match_id": match_id}, {"match_id": match_id, **scouting_doc}, upsert=True)
        done.append("scouting")

        FULL_ANALYSIS_STATE[match_id] = {
            "status": "done", "stage": "done", "stages_done": done,
            "total_stages": len(FULL_STAGES),
            "tagged_outcomes": len(outcomes),
        }
        logger.info(f"Full analysis done for {match_id}")
    except Exception as e:
        logger.error(f"Full analysis failed for {match_id} at stage "
                     f"{FULL_ANALYSIS_STATE.get(match_id, {}).get('stage')}: {e}", exc_info=True)
        FULL_ANALYSIS_STATE[match_id] = {
            "status": "failed", "stages_done": done, "error": str(e),
            "stage": FULL_ANALYSIS_STATE.get(match_id, {}).get("stage"),
        }


@api_router.post("/analysis/full/{match_id}")
async def start_full_analysis(match_id: str):
    """One-click: run rallies → court control → shot patterns → scouting in sequence."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(503, "Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(404, "Match not found")
    if FULL_ANALYSIS_STATE.get(match_id, {}).get("status") == "running":
        return {"message": "Already running", "state": FULL_ANALYSIS_STATE[match_id]}

    video_path = str(UPLOAD_DIR / match["video_filename"])
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = total / fps if fps else 300.0

    FULL_ANALYSIS_STATE[match_id] = {"status": "running", "stage": "rallies",
                                     "stage_label": FULL_STAGE_LABELS["rallies"],
                                     "stages_done": [], "total_stages": len(FULL_STAGES)}
    asyncio.create_task(
        _run_full_analysis(match_id, match, video_path, 0.0, duration_s))
    return {"message": "Full analysis started", "total_stages": len(FULL_STAGES)}


@api_router.get("/analysis/full/{match_id}")
async def get_full_analysis_status(match_id: str):
    """Progress of the one-click full analysis."""
    return FULL_ANALYSIS_STATE.get(match_id, {"status": "idle"})


@api_router.post("/matches/{match_id}/start-analysis")
async def start_analysis(match_id: str, background_tasks: BackgroundTasks):
    """Start analysis without player selection"""
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    if match.get("status") in ["processing", "completed"]:
        return {"message": "Analysis already started"}
    
    await db.matches.update_one(
        {"id": match_id},
        {"$set": {"status": "processing", "progress": 5}}
    )
    
    video_path = UPLOAD_DIR / match["video_filename"]
    background_tasks.add_task(process_video_analysis, match_id, str(video_path))
    
    return {"message": "Analysis started"}

@api_router.get("/matches/{match_id}/export/json")
async def export_match_json(match_id: str):
    """Export match analysis as JSON"""
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Remove thumbnail from export (too large)
    export_data = {k: v for k, v in match.items() if k != "thumbnail"}
    
    return StreamingResponse(
        io.BytesIO(json.dumps(export_data, indent=2, default=str).encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=squashsense_{match_id}.json"}
    )

@api_router.get("/matches/{match_id}/export/pdf")
async def export_match_pdf(match_id: str):
    """Export match analysis as PDF report"""
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph(f"SquashSense AI Analysis Report", title_style))
    story.append(Paragraph(f"Match: {match.get('title', 'Untitled')}", styles['Heading2']))
    story.append(Spacer(1, 20))
    
    # Summary stats
    story.append(Paragraph("Match Summary", styles['Heading2']))
    summary_data = [
        ["Total Shots", str(match.get("total_shots", 0))],
        ["Total Rallies", str(match.get("total_rallies", 0))],
        ["Duration", f"{match.get('duration', 0):.1f} seconds"],
        ["Status", match.get("status", "unknown")]
    ]
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Shot distribution
    shot_dist = match.get("shot_distribution", {})
    if shot_dist:
        story.append(Paragraph("Shot Distribution", styles['Heading2']))
        shot_data = [[shot_type.capitalize(), str(count)] for shot_type, count in shot_dist.items()]
        shot_table = Table([["Shot Type", "Count"]] + shot_data, colWidths=[200, 200])
        shot_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(shot_table)
        story.append(Spacer(1, 20))
    
    # Key insights
    insights = match.get("key_insights", [])
    if insights:
        story.append(Paragraph("Key Insights", styles['Heading2']))
        for insight in insights:
            story.append(Paragraph(f"• {insight}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=squashsense_{match_id}.pdf"}
    )

# ===================== TRAINING DATA ENDPOINTS =====================

class CorrectionRequest(BaseModel):
    shot_index: int
    corrected_shot_type: str
    corrected_player: str

@api_router.post("/matches/{match_id}/correct-shot")
async def correct_shot(match_id: str, correction: CorrectionRequest):
    """Submit a shot correction - this data is used for training"""
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    shots = match.get("shots", [])
    if correction.shot_index >= len(shots):
        raise HTTPException(status_code=400, detail="Invalid shot index")
    
    original_shot = shots[correction.shot_index]
    
    # Store correction for training
    correction_doc = {
        "id": str(uuid.uuid4()),
        "match_id": match_id,
        "shot_index": correction.shot_index,
        "original_shot_type": original_shot.get("shot_type"),
        "corrected_shot_type": correction.corrected_shot_type,
        "original_player": original_shot.get("player"),
        "corrected_player": correction.corrected_player,
        "timestamp": original_shot.get("timestamp"),
        "pose_data": original_shot.get("pose_data"),
        "corrected_at": datetime.now(timezone.utc).isoformat(),
        "verified": False
    }
    
    await db.training_corrections.insert_one(correction_doc)
    
    # Update the shot in the match
    shots[correction.shot_index]["shot_type"] = correction.corrected_shot_type
    shots[correction.shot_index]["player"] = correction.corrected_player
    shots[correction.shot_index]["user_corrected"] = True
    
    # Recalculate shot distribution
    shot_distribution = {"drive": 0, "drop": 0, "boast": 0, "volley": 0, "lob": 0, "kill": 0, "serve": 0}
    for shot in shots:
        st = shot.get("shot_type", "drive")
        if st in shot_distribution:
            shot_distribution[st] += 1
    
    await db.matches.update_one(
        {"id": match_id},
        {"$set": {"shots": shots, "shot_distribution": shot_distribution}}
    )
    
    return {"message": "Correction saved", "correction_id": correction_doc["id"]}

@api_router.get("/training/flywheel")
async def get_flywheel_summary():
    """The data flywheel at a glance — every human-provided label across the library
    that can train the models (the moat). Layer 7 of the architecture.

    Aggregates: ball position labels, rally-outcome tags, player identities, and
    shot-type corrections. These are what scheduled retraining consumes; growth
    here = the system getting smarter over time.
    """
    ball_labels = await db.ball_labels.count_documents({})
    shot_corrections = await db.training_corrections.count_documents({})
    verified_corrections = await db.training_corrections.count_documents({"verified": True})

    # Rally outcomes tagged across all matches
    outcome_tags = 0
    matches_with_outcomes = 0
    async for seg in db.rally_segments.find({}, {"outcomes": 1}):
        n = len(seg.get("outcomes", {}) or {})
        if n:
            matches_with_outcomes += 1
            outcome_tags += n

    identified = await db.matches.count_documents(
        {"player1_color_sig": {"$exists": True, "$ne": None}})
    calibrated = await db.matches.count_documents(
        {"court_calibration": {"$exists": True, "$ne": None}})

    total_labels = ball_labels + shot_corrections + outcome_tags
    return {
        "total_human_labels": total_labels,
        "by_source": {
            "ball_position_labels": ball_labels,
            "rally_outcome_tags": outcome_tags,
            "shot_type_corrections": shot_corrections,
        },
        "context": {
            "matches_with_outcomes": matches_with_outcomes,
            "matches_player_identified": identified,
            "matches_calibrated": calibrated,
            "verified_shot_corrections": verified_corrections,
        },
        # What each label source trains, and the retrain trigger.
        "trains": {
            "ball_position_labels": "tracknet ball model (retrain when +200 new manual labels)",
            "rally_outcome_tags": "outcome ground-truth for rules-engine eval + shot/error stats",
            "shot_type_corrections": "shot classifier (Layer 5) — once ≥ ~500 labels, train to replace heuristics",
        },
        "drift_guard": "human-confirmed labels outrank model pseudo-labels; cap self-labels per run",
    }


def _eval_ball_sync(samples: List[Dict], setup: str) -> Dict:
    """Held-out eval: run the ball detector on labelled frames and measure
    localisation error. Samples are [{video_path, frame_index, x, y, native_w}]."""
    import cv2
    from perception.ball import get_ball_detector
    det = get_ball_detector(setup)
    errors, detected, total = [], 0, 0
    # group by video to limit reopening
    by_vid: Dict[str, List[Dict]] = {}
    for s in samples:
        by_vid.setdefault(s["video_path"], []).append(s)
    for vid, items in by_vid.items():
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        scale = min(1.0, 1280 / nw)
        for s in items:
            fi = s["frame_index"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fi - 2))
            win = []
            for _ in range(5):
                ok, f = cap.read()
                if not ok:
                    break
                h, w = f.shape[:2]
                if w > 1280:
                    f = cv2.resize(f, (1280, int(h * 1280 / w)))
                win.append(f)
            if len(win) < 3:
                continue
            total += 1
            per = det.detect_window(win, fi - 2, fps)
            # detection at the labelled frame (index 2 in the window)
            cands = per[2] if len(per) > 2 else []
            if cands:
                detected += 1
                lx, ly = s["x"] * scale, s["y"] * scale
                errors.append(float(np.hypot(cands[0].x - lx, cands[0].y - ly)))
        cap.release()
    errors_sorted = sorted(errors)
    median = errors_sorted[len(errors_sorted) // 2] if errors_sorted else None
    return {
        "n_eval": total,
        "detection_rate": round(detected / total, 3) if total else 0,
        "median_error_px": round(median, 2) if median is not None else None,
        "mean_error_px": round(float(np.mean(errors)), 2) if errors else None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def _eval_continuity_sync(windows: List[Dict], setup: str) -> Dict:
    """Measure ball-track CONTINUITY (the 3D-timeline bottleneck) by running the
    detector+tracker over labelled rally windows and reporting how fragmented the
    track is. Better model ⇒ longer continuous arcs, fewer fragments.

    windows: [{video_path, start_f, n_frames}]
    """
    import cv2
    from perception.ball import get_ball_detector, BallTracker
    det = get_ball_detector(setup)
    arc_lengths, n_arcs_list, detect_rates = [], [], []
    for w in windows:
        cap = cv2.VideoCapture(w["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, w["start_f"])
        frames = []
        for _ in range(w["n_frames"]):
            ok, f = cap.read()
            if not ok:
                break
            h, wd = f.shape[:2]
            if wd > 1280:
                f = cv2.resize(f, (1280, int(h * 1280 / wd)))
            frames.append(f)
        cap.release()
        if len(frames) < 15:
            continue
        per = det.detect_window(frames, w["start_f"], fps)
        detect_rates.append(sum(1 for c in per if c) / len(per))
        tracker = BallTracker()
        tracks = tracker.build_tracks(per)
        qual = [t for t in tracks if tracker._track_quality(t) > 0 and len(t.points) >= 4]
        if qual:
            lens = [len(t.points) for t in qual]
            arc_lengths.append(float(np.mean(lens)))
            n_arcs_list.append(len(qual))
    if not arc_lengths:
        return {"continuity": None}
    return {
        "mean_arc_frames": round(float(np.mean(arc_lengths)), 1),
        "longest_arc_frames": None,
        "avg_fragments_per_window": round(float(np.mean(n_arcs_list)), 1),
        "detection_rate": round(float(np.mean(detect_rates)), 3),
        "windows_evaluated": len(arc_lengths),
    }


@api_router.post("/training/eval-ball")
async def eval_ball_model(sample_size: int = 24):
    """Evaluate the ball model on a held-out sample of human ball labels.
    Tracks accuracy over time so retraining never silently regresses."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(503, "Perception spine unavailable")
    # Gather labelled points with their video paths
    samples: List[Dict] = []
    async for doc in db.ball_labels.find({"num_points": {"$gt": 0}}):
        match = await db.matches.find_one({"id": doc["match_id"]}, {"video_filename": 1})
        if not match or not match.get("video_filename"):
            continue
        vp = str(UPLOAD_DIR / match["video_filename"])
        for p in (doc.get("points") or []):
            if "frame_index" in p and "x" in p and "y" in p:
                samples.append({"video_path": vp, "frame_index": int(p["frame_index"]),
                                "x": float(p["x"]), "y": float(p["y"])})
    if not samples:
        return {"error": "No ball labels to evaluate."}
    # Held-out sample (deterministic: every Nth point)
    step = max(1, len(samples) // sample_size)
    held = samples[::step][:sample_size]
    result = await asyncio.to_thread(_eval_ball_sync, held, "phone")
    result["total_labels_available"] = len(samples)

    # Continuity: run the tracker over a few labelled rally windows.
    by_vid: Dict[str, List[int]] = {}
    for s in samples:
        by_vid.setdefault(s["video_path"], []).append(s["frame_index"])
    windows = []
    for vp, fis in list(by_vid.items())[:3]:
        fis.sort()
        start_f = max(0, fis[len(fis) // 2] - 120)   # centre on the labels
        windows.append({"video_path": vp, "start_f": start_f, "n_frames": 300})
    if windows:
        cont = await asyncio.to_thread(_eval_continuity_sync, windows, "phone")
        result["continuity"] = cont

    await db.model_evals.insert_one({"kind": "ball", **result})
    return result


@api_router.get("/training/eval-history")
async def eval_history(kind: str = "ball", limit: int = 20):
    """Past eval scores (newest first) — the no-regression record."""
    docs = await db.model_evals.find({"kind": kind}, {"_id": 0}).sort(
        "evaluated_at", -1).to_list(length=limit)
    return {"kind": kind, "evals": docs}


@api_router.get("/training/stats")
async def get_training_stats():
    """Get training data statistics"""
    total_corrections = await db.training_corrections.count_documents({})
    
    # Get corrections by shot type
    pipeline = [
        {"$group": {"_id": "$corrected_shot_type", "count": {"$sum": 1}}}
    ]
    corrections_by_type = {}
    async for doc in db.training_corrections.aggregate(pipeline):
        corrections_by_type[doc["_id"]] = doc["count"]
    
    verified_count = await db.training_corrections.count_documents({"verified": True})
    
    # Estimate accuracy based on correction rate
    total_shots = await db.matches.aggregate([
        {"$group": {"_id": None, "total": {"$sum": "$total_shots"}}}
    ]).to_list(1)
    total_shots_count = total_shots[0]["total"] if total_shots else 0
    
    accuracy_estimate = 1.0 - (total_corrections / max(1, total_shots_count))
    
    return {
        "total_corrections": total_corrections,
        "corrections_by_shot_type": corrections_by_type,
        "verified_samples": verified_count,
        "model_accuracy_estimate": round(max(0, accuracy_estimate) * 100, 1),
        "total_shots_analyzed": total_shots_count,
        "training_ready": total_corrections >= 100  # Need at least 100 corrections
    }

@api_router.get("/training/export")
async def export_training_data():
    """Export training data for model fine-tuning"""
    corrections = await db.training_corrections.find({}, {"_id": 0}).to_list(10000)
    
    # Format for training
    training_data = []
    for c in corrections:
        training_data.append({
            "shot_type": c.get("corrected_shot_type"),
            "player": c.get("corrected_player"),
            "timestamp": c.get("timestamp"),
            "pose_data": c.get("pose_data"),
            "original_prediction": c.get("original_shot_type")
        })
    
    return {
        "count": len(training_data),
        "data": training_data,
        "export_time": datetime.now(timezone.utc).isoformat()
    }

@api_router.get("/matches/{match_id}/shots-with-frames")
async def get_shots_with_frames(match_id: str):
    """Get shots with their frame data for correction UI"""
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Return shots with pose data
    return {
        "match_id": match_id,
        "shots": match.get("shots", []),
        "total_shots": match.get("total_shots", 0)
    }

# ===================== BALL ANNOTATION ENDPOINTS =====================
# Bootstrap the labelled squash-ball dataset that a future TrackNet will train on.
# Flow: extract ranked candidate ball-tracks from a video window -> human confirms
# /rejects each track in the review UI -> confirmed tracks become ground-truth ball
# positions -> export for training.

class BallExtractRequest(BaseModel):
    start_s: float = 0.0
    duration_s: float = 8.0
    max_tracks: int = 12


@api_router.post("/annotation/ball/extract/{match_id}")
async def extract_ball_candidates(match_id: str, req: BallExtractRequest):
    """Run the classical detector over a window and store candidate ball-tracks."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Perception spine unavailable")

    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    video_path = str(UPLOAD_DIR / match["video_filename"])
    calib_dict = match.get("court_calibration")
    calibration = None
    if calib_dict:
        cap = cv2.VideoCapture(video_path)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
        cap.release()
        calibration = CourtCalibration.from_normalized(calib_dict, vw, vh)

    result = await asyncio.to_thread(
        extract_candidate_tracks,
        video_path,
        req.start_s,
        req.duration_s,
        req.max_tracks,
        "yolo11n.pt",
        calibration,
    )

    task_doc = {
        "id": str(uuid.uuid4()),
        "match_id": match_id,
        "start_s": req.start_s,
        "duration_s": req.duration_s,
        "fps": result.get("fps"),
        "num_tracks": result.get("num_tracks", 0),
        "tracks": result.get("tracks", []),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.ball_anno_tasks.insert_one(task_doc.copy())

    task_doc.pop("_id", None)
    return task_doc


@api_router.get("/annotation/ball/tasks/{match_id}")
async def get_ball_tasks(match_id: str):
    """Return extracted candidate-track tasks for a match (for the review UI)."""
    tasks = await db.ball_anno_tasks.find(
        {"match_id": match_id}, {"_id": 0}
    ).sort("created_at", -1).to_list(50)
    # Attach any existing labels so the UI reflects prior decisions on revisit.
    for task in tasks:
        labels = await db.ball_labels.find(
            {"task_id": task["id"]}, {"_id": 0, "track_id": 1, "label": 1}
        ).to_list(1000)
        task["labels"] = labels
    return {"match_id": match_id, "tasks": tasks}


class BallLabelRequest(BaseModel):
    task_id: str
    track_id: int
    label: str  # "ball" | "not_ball" | "unsure"
    # Optional human-corrected points; if omitted the track's points are used as-is.
    points: Optional[List[Dict[str, Any]]] = None


@api_router.post("/annotation/ball/label")
async def label_ball_track(match_id: str, req: BallLabelRequest):
    """Record a human label for a candidate track. 'ball' => ground-truth positions."""
    if req.label not in ("ball", "not_ball", "unsure"):
        raise HTTPException(status_code=400, detail="Invalid label")

    task = await db.ball_anno_tasks.find_one(
        {"id": req.task_id, "match_id": match_id}, {"_id": 0}
    )
    if not task:
        raise HTTPException(status_code=404, detail="Annotation task not found")

    points = req.points
    if points is None:
        track = next((t for t in task["tracks"] if t["track_id"] == req.track_id), None)
        if track is None:
            raise HTTPException(status_code=404, detail="Track not found in task")
        # Strip the crop images from stored ground truth (keep it lean).
        points = [
            {"frame_index": p["frame_index"], "timestamp": p["timestamp"],
             "x": p["x"], "y": p["y"]}
            for p in track["points"]
        ]

    label_doc = {
        "id": str(uuid.uuid4()),
        "match_id": match_id,
        "task_id": req.task_id,
        "track_id": req.track_id,
        "label": req.label,
        "points": points if req.label == "ball" else [],
        "num_points": len(points) if req.label == "ball" else 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    # One label per (task, track): replace if re-labelled.
    await db.ball_labels.replace_one(
        {"match_id": match_id, "task_id": req.task_id, "track_id": req.track_id},
        label_doc,
        upsert=True,
    )
    return {"message": "Label saved", "label": req.label, "ball_points": label_doc["num_points"]}


@api_router.get("/annotation/ball/frames/{match_id}")
async def get_ball_marking_frames(
    match_id: str, start_s: float = 0.0, count: int = 24, step: int = 2
):
    """Return a strip of frames so a human can manually mark the ball position
    when no auto-detected candidate track is the ball."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    video_path = str(UPLOAD_DIR / match["video_filename"])
    result = await asyncio.to_thread(
        extract_frames_for_marking, video_path, start_s, count, step
    )
    return result


class ManualBallPoint(BaseModel):
    frame_index: int
    nx: float  # normalized 0..1 image fractions
    ny: float


class ManualBallLabelRequest(BaseModel):
    points: List[ManualBallPoint]
    native_width: int
    native_height: int


class PropagateRequest(BaseModel):
    start_frame_index: int
    nx: float  # normalized 0..1 click position
    ny: float
    n_frames: int = 30


@api_router.post("/annotation/ball/propagate/{match_id}")
async def propagate_ball_click(match_id: str, req: PropagateRequest):
    """Click the ball once → optical flow tracks it across the next N frames.

    Returns the auto-tracked points (with review crops) for the human to verify
    and save. Turns ~30 clicks into 1 click + a glance."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    video_path = str(UPLOAD_DIR / match["video_filename"])
    cap = cv2.VideoCapture(video_path)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
    cap.release()

    result = await asyncio.to_thread(
        propagate_ball, video_path, req.start_frame_index,
        req.nx * vw, req.ny * vh, req.n_frames,
    )
    # add normalized coords so the UI can save directly
    for p in result.get("points", []):
        p["nx"] = round(p["x"] / vw, 4)
        p["ny"] = round(p["y"] / vh, 4)
    result["native_width"] = vw
    result["native_height"] = vh
    return result


@api_router.post("/annotation/ball/manual-label")
async def manual_ball_label(match_id: str, req: ManualBallLabelRequest):
    """Store human-marked ball positions (highest-quality ground truth).

    Coordinates arrive normalized; we convert to native pixels using the frame
    size the UI reported. Stored in the same ball_labels collection (label='ball',
    source='manual') so they flow into the same training export.
    """
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if not req.points:
        raise HTTPException(status_code=400, detail="No points provided")

    points = [
        {
            "frame_index": p.frame_index,
            "x": round(p.nx * req.native_width, 1),
            "y": round(p.ny * req.native_height, 1),
        }
        for p in req.points
    ]
    label_doc = {
        "id": str(uuid.uuid4()),
        "match_id": match_id,
        "task_id": "manual",
        "track_id": -1,
        "label": "ball",
        "source": "manual",
        "points": points,
        "num_points": len(points),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.ball_labels.insert_one(label_doc.copy())
    return {"message": "Manual labels saved", "ball_points": len(points)}


@api_router.get("/annotation/ball/dataset")
async def export_ball_dataset():
    """Export confirmed ball positions as a flat training set for TrackNet."""
    confirmed = await db.ball_labels.find(
        {"label": "ball"}, {"_id": 0}
    ).to_list(100000)

    samples = []
    for doc in confirmed:
        for p in doc.get("points", []):
            samples.append({
                "match_id": doc["match_id"],
                "frame_index": p["frame_index"],
                "x": p["x"],
                "y": p["y"],
            })

    total_labeled = await db.ball_labels.count_documents({})
    return {
        "ball_tracks_confirmed": len(confirmed),
        "ball_points": len(samples),
        "tracks_labeled_total": total_labeled,
        "samples": samples,
        "export_time": datetime.now(timezone.utc).isoformat(),
    }


@api_router.get("/annotation/ball/stats")
async def ball_annotation_stats():
    """Progress of the ball-labelling effort."""
    by_label = {}
    async for d in db.ball_labels.aggregate([
        {"$group": {"_id": "$label", "count": {"$sum": 1}}}
    ]):
        by_label[d["_id"]] = d["count"]
    ball_points = await db.ball_labels.aggregate([
        {"$match": {"label": "ball"}},
        {"$group": {"_id": None, "pts": {"$sum": "$num_points"}}},
    ]).to_list(1)
    models = {
        "phone": os.path.exists(_domain_weights_path("phone")),
        "broadcast": os.path.exists(_domain_weights_path("broadcast")),
        "combined": os.path.exists(BALL_WEIGHTS_PATH),
    }
    return {
        "tracks_labeled_by_type": by_label,
        "confirmed_ball_points": ball_points[0]["pts"] if ball_points else 0,
        "tasks_extracted": await db.ball_anno_tasks.count_documents({}),
        "model_active": any(models.values()),
        "models_active": models,
        "recommended_min_points": 200,
    }


# ----- TrackNet training, triggered from the UI -----
WEIGHTS_DIR = ROOT_DIR / "perception" / "weights"
# A specialised model per camera SETUP — routed at detection time.
BALL_WEIGHTS_PATH = str(WEIGHTS_DIR / "tracknet.pt")  # legacy combined (fallback)
# In-process training state (single uvicorn worker). Polled by the UI.
BALL_TRAINING_STATE: Dict[str, Any] = {"status": "idle"}


def _domain_weights_path(domain: str) -> str:
    return str(WEIGHTS_DIR / f"tracknet_{domain}.pt")


def _match_setup(match: Dict[str, Any]) -> str:
    """Camera setup of a video: broadcast (pro wide angle) vs phone (close games)."""
    return "broadcast" if match.get("source") == "youtube" else "phone"


async def _run_ball_training(samples: List[Dict], video_map: Dict[str, str],
                             epochs: int, min_samples: int, out_weights: str,
                             domain: str = "phone"):
    from perception.tracknet import train_tracknet

    def progress_cb(ep, total, loss):
        BALL_TRAINING_STATE.update(
            {"status": "running", "epoch": ep, "epochs": total, "loss": round(loss, 5)}
        )

    def resolver(mid: str) -> str:
        return str(UPLOAD_DIR / video_map[mid])

    BALL_TRAINING_STATE.clear()
    BALL_TRAINING_STATE.update({"status": "running", "epoch": 0, "epochs": epochs,
                                "samples": len(samples), "domain": domain})
    try:
        # Frozen ImageNet encoder + trained decoder: stable and memory-safe.
        report = await asyncio.to_thread(
            train_tracknet, samples, resolver, out_weights,
            epochs, 8, 1e-3, (288, 512), None, min_samples, progress_cb,
            True, True,  # pretrained=True, freeze_encoder=True
        )
        report["domain"] = domain
        BALL_TRAINING_STATE.clear()
        BALL_TRAINING_STATE.update({"status": "done", "domain": domain, "report": report,
                                    "model_active": os.path.exists(out_weights)})
        logger.info(f"Ball training finished ({domain}): {report}")
    except Exception as e:
        logger.error(f"Ball training failed: {e}")
        BALL_TRAINING_STATE.clear()
        BALL_TRAINING_STATE.update({"status": "failed", "error": str(e)})


class TrainBallRequest(BaseModel):
    epochs: int = 20
    min_samples: int = 30  # UI lets you train early to see it work; 200+ is good
    domain: str = "phone"   # which specialised model to train: "phone" | "broadcast"


@api_router.post("/annotation/ball/train")
async def train_ball_model(req: TrainBallRequest):
    """Train the ball model for one camera setup (phone or broadcast), on that
    setup's labels only — keeps each model specialised. Runs in background."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Perception spine unavailable")
    if req.domain not in ("phone", "broadcast"):
        raise HTTPException(status_code=400, detail="domain must be phone|broadcast")
    if BALL_TRAINING_STATE.get("status") == "running":
        return {"message": "Training already running", "state": BALL_TRAINING_STATE}

    broadcast_ids = {mm["id"] async for mm in
                     db.matches.find({"source": "youtube"}, {"_id": 0, "id": 1})}

    def _in_domain(mid: str) -> bool:
        is_broadcast = mid in broadcast_ids
        return is_broadcast if req.domain == "broadcast" else not is_broadcast

    # Gather only this setup's labels, split verified (manual) vs self-trained.
    manual: List[Dict] = []
    selftrain: List[Dict] = []
    video_map: Dict[str, str] = {}
    async for doc in db.ball_labels.find({"label": "ball"}, {"_id": 0}):
        mid = doc["match_id"]
        if mid not in video_map:
            m = await db.matches.find_one({"id": mid}, {"_id": 0, "video_filename": 1})
            video_map[mid] = m["video_filename"] if m else None
        if not video_map[mid] or not _in_domain(mid):
            continue
        bucket = selftrain if doc.get("source") == "selftrain" else manual
        for p in doc.get("points", []):
            bucket.append({"match_id": mid, "frame_index": p["frame_index"],
                           "x": p["x"], "y": p["y"]})

    # Drift guard: self-trained pseudo-labels ≤ verified manual labels in a run.
    capped_selftrain = selftrain
    if len(manual) > 0 and len(selftrain) > len(manual):
        random.shuffle(selftrain)
        capped_selftrain = selftrain[: len(manual)]
    samples = manual + capped_selftrain

    if len(samples) < req.min_samples:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(samples)} {req.domain} ball points — need >= "
                   f"{req.min_samples}. Label more {req.domain} footage.",
        )

    out_weights = _domain_weights_path(req.domain)
    asyncio.create_task(
        _run_ball_training(samples, video_map, req.epochs, req.min_samples,
                           out_weights, req.domain)
    )
    return {
        "message": f"Training {req.domain} model started",
        "domain": req.domain, "samples": len(samples), "epochs": req.epochs,
        "manual": len(manual), "selftrain_used": len(capped_selftrain),
    }


@api_router.get("/annotation/ball/train/status")
async def ball_train_status():
    """Poll training progress."""
    return BALL_TRAINING_STATE


# ===================== TRAINING VIDEO LIBRARY =====================
# Bulk-ingest a library of past-game videos as training sources. These become
# matches with source="library" so they flow through the same ball-labelling and
# training tools; training (POST /annotation/ball/train) already aggregates
# labelled points across ALL videos.

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


async def _register_video_file(src_path: Path, title: str) -> Optional[str]:
    """Place a video into uploads/ (hardlink if possible, else copy) and create a
    library match record. Returns the match id, or None on failure."""
    try:
        ext = src_path.suffix.lower() or ".mp4"
        unique = f"{uuid.uuid4()}{ext}"
        dest = UPLOAD_DIR / unique
        try:
            os.link(src_path, dest)          # instant, no extra disk
        except Exception:
            import shutil
            shutil.copy2(src_path, dest)

        thumbnail = await generate_thumbnail(str(dest))
        match = MatchAnalysis(title=title, video_filename=unique, status="library")
        doc = match.model_dump()
        doc["upload_time"] = doc["upload_time"].isoformat()
        doc["source"] = "library"
        if thumbnail:
            doc["thumbnail"] = thumbnail
        await db.matches.insert_one(doc)
        return match.id
    except Exception as e:
        logger.error(f"Failed to register {src_path}: {e}")
        return None


class IngestFolderRequest(BaseModel):
    folder_path: str


@api_router.post("/training/ingest")
async def ingest_video_folder(req: IngestFolderRequest):
    """Register every video in a server-side folder as a training-library source."""
    folder = Path(req.folder_path).expanduser()
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a folder: {folder}")

    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not files:
        raise HTTPException(status_code=400, detail="No video files found in folder")

    ingested = []
    for f in files:
        mid = await _register_video_file(f, f.stem)
        if mid:
            ingested.append({"id": mid, "title": f.stem, "file": f.name})
    return {"ingested": len(ingested), "videos": ingested}


# ----- Ingest from YouTube (background download) -----
YT_INGEST_STATE: Dict[str, Any] = {"status": "idle"}


def _download_youtube(url: str, dest_path: Path) -> Dict[str, Any]:
    """Download a single YouTube video to dest_path (progressive mp4, no ffmpeg)."""
    import yt_dlp

    opts = {
        # We only need frames (no audio), so prefer a high-res VIDEO-ONLY stream
        # (up to 1080p) — single file, no ffmpeg merge. Fall back to progressive.
        "format": (
            "bestvideo[ext=mp4][height<=1080]/"
            "best[acodec!=none][vcodec!=none][height<=720]/best"
        ),
        "outtmpl": str(dest_path),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return {"title": info.get("title", "YouTube video"),
            "duration": info.get("duration", 0)}


async def _run_youtube_ingest(urls: List[str]):
    YT_INGEST_STATE.clear()
    YT_INGEST_STATE.update({"status": "running", "processed": 0,
                            "total": len(urls), "added": 0, "errors": []})
    added = 0
    for i, url in enumerate(urls):
        try:
            unique = f"{uuid.uuid4()}.mp4"
            dest = UPLOAD_DIR / unique
            meta = await asyncio.to_thread(_download_youtube, url, dest)
            thumbnail = await generate_thumbnail(str(dest))
            match = MatchAnalysis(title=meta["title"][:120],
                                  video_filename=unique, status="library")
            doc = match.model_dump()
            doc["upload_time"] = doc["upload_time"].isoformat()
            doc["source"] = "youtube"
            doc["youtube_url"] = url
            if thumbnail:
                doc["thumbnail"] = thumbnail
            await db.matches.insert_one(doc)
            added += 1
        except Exception as e:
            logger.error(f"YouTube ingest failed for {url}: {e}")
            YT_INGEST_STATE["errors"].append({"url": url, "error": str(e)[:160]})
        YT_INGEST_STATE.update({"processed": i + 1, "added": added})
    YT_INGEST_STATE.update({"status": "done", "added": added})


class YoutubeIngestRequest(BaseModel):
    urls: List[str]


@api_router.post("/training/ingest-youtube")
async def ingest_youtube(req: YoutubeIngestRequest):
    """Download YouTube videos and add them to the training library (background)."""
    urls = [u.strip() for u in req.urls if u.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    if YT_INGEST_STATE.get("status") == "running":
        return {"message": "An ingest is already running", "state": YT_INGEST_STATE}
    asyncio.create_task(_run_youtube_ingest(urls))
    return {"message": "Downloading in background", "count": len(urls)}


@api_router.get("/training/ingest-youtube/status")
async def ingest_youtube_status():
    return YT_INGEST_STATE


@api_router.get("/training/library")
async def get_training_library():
    """List all videos with per-video labelling progress (for the library UI)."""
    matches = await db.matches.find(
        {}, {"_id": 0, "id": 1, "title": 1, "status": 1, "source": 1,
             "thumbnail": 1, "duration": 1, "upload_time": 1}
    ).sort("upload_time", -1).to_list(1000)

    # Per-video confirmed ball points.
    points_by_match = {}
    async for d in db.ball_labels.aggregate([
        {"$match": {"label": "ball"}},
        {"$group": {"_id": "$match_id", "pts": {"$sum": "$num_points"}}},
    ]):
        points_by_match[d["_id"]] = d["pts"]

    for m in matches:
        m["ball_points"] = points_by_match.get(m["id"], 0)

    total_points = sum(points_by_match.values())
    return {
        "videos": matches,
        "video_count": len(matches),
        "total_ball_points": total_points,
        "model_active": os.path.exists(BALL_WEIGHTS_PATH),
    }


# ===================== RALLY SEGMENTATION =====================
# Logic (no training): segment a span into rallies from sustained ball motion.
RALLY_STATE: Dict[str, Any] = {}   # match_id -> {status, ...}
RALLY_CLIPS_DIR = ROOT_DIR / "rally_clips"
RALLY_CLIPS_DIR.mkdir(exist_ok=True)


async def _load_calibration(match: dict, video_path: str):
    """Load and return a CourtCalibration for a match, or None if uncalibrated."""
    calib_dict = match.get("court_calibration")
    if not calib_dict or not CourtCalibration:
        return None
    import cv2
    cap = cv2.VideoCapture(video_path)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    try:
        return CourtCalibration.from_normalized(calib_dict, vw, vh)
    except Exception:
        return None


def _load_ref_sigs(match: dict):
    """Return {1: sig, 2: sig} colour signatures from named players, or None."""
    s1 = match.get("player1_color_sig")
    s2 = match.get("player2_color_sig")
    if s1 and s2:
        return {1: s1, 2: s2}
    return None


async def _run_rally_segmentation(match_id: str, video_path: str,
                                  start_s: float, duration_s: float, setup: str,
                                  calibration=None):
    RALLY_STATE[match_id] = {"status": "running"}
    try:
        # AUDIO is the primary signal — the ball-strike "thwack" segments rallies
        # far more reliably than visual ball tracking (and in ~0.5s). Fall back to
        # the visual segmenters only if there's no audio track.
        result = None
        if segment_rallies_audio is not None:
            audio_res = await asyncio.to_thread(
                segment_rallies_audio, video_path, start_s, duration_s
            )
            if audio_res.get("rallies"):
                result = audio_res
        if result is None:
            if segment_rallies_v2 is not None:
                result = await asyncio.to_thread(
                    segment_rallies_v2, video_path, start_s, duration_s, setup, calibration
                )
            else:
                result = await asyncio.to_thread(
                    segment_rallies, video_path, start_s, duration_s, setup
                )
        result["status"] = "done"
        result["start_s"] = start_s
        RALLY_STATE[match_id] = result
        await db.rally_segments.replace_one(
            {"match_id": match_id},
            {"match_id": match_id, "start_s": start_s, "result": result,
             "created_at": datetime.now(timezone.utc).isoformat()},
            upsert=True,
        )
        logger.info(f"Rally segmentation done for {match_id}: {result['num_rallies']} rallies "
                    f"(method={result.get('method','v1')})")
        # Extract a short video clip for each rally so the user can review boundaries.
        clip_dir = RALLY_CLIPS_DIR / match_id
        clip_dir.mkdir(exist_ok=True)
        for r in result.get("rallies", []):
            out = str(clip_dir / f"rally_{r['rally_id']}.mp4")
            await asyncio.to_thread(
                extract_rally_clip, video_path, r["start_t"], r["end_t"], out
            )
        logger.info(f"Rally clips extracted for {match_id}")
    except Exception as e:
        logger.error(f"Rally segmentation failed for {match_id}: {e}", exc_info=True)
        RALLY_STATE[match_id] = {"status": "failed", "error": str(e)}


class RallySegRequest(BaseModel):
    start_s: float = 0.0
    duration_s: float = 60.0


@api_router.post("/analysis/rallies/{match_id}")
async def segment_match_rallies(match_id: str, req: RallySegRequest):
    """Segment a video span into rallies (start/end + shot counts). Background."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if RALLY_STATE.get(match_id, {}).get("status") == "running":
        return {"message": "Already segmenting", "state": RALLY_STATE[match_id]}
    video_path = str(UPLOAD_DIR / match["video_filename"])
    calibration = await _load_calibration(match, video_path)
    asyncio.create_task(
        _run_rally_segmentation(match_id, video_path, req.start_s, req.duration_s,
                                _match_setup(match), calibration)
    )
    method = "dual_signal_v2" if segment_rallies_v2 else "motion_v1"
    return {"message": f"Segmenting rallies ({method})", "match_id": match_id}


@api_router.get("/analysis/rallies/{match_id}")
async def get_match_rallies(match_id: str):
    """Live status + stored outcomes."""
    state = RALLY_STATE.get(match_id)
    doc = await db.rally_segments.find_one({"match_id": match_id}, {"_id": 0})
    outcomes = doc.get("outcomes", {}) if doc else {}
    if state:
        return {**state, "outcomes": outcomes}
    if doc:
        return {**doc["result"], "status": "done", "outcomes": outcomes}
    return {"status": "idle"}


@api_router.get("/analysis/rallies/{match_id}/clip/{rally_id}")
async def get_rally_clip(match_id: str, rally_id: int):
    """Stream the extracted video clip for a single rally."""
    from fastapi.responses import FileResponse
    path = RALLY_CLIPS_DIR / match_id / f"rally_{rally_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clip not ready yet")
    return FileResponse(str(path), media_type="video/mp4",
                        headers={"Cache-Control": "public, max-age=3600"})


class RallyOutcomeRequest(BaseModel):
    outcome: str  # "p1" | "p2" | "let" | "stroke_p1" | "stroke_p2"


@api_router.post("/analysis/rallies/{match_id}/{rally_id}/outcome")
async def set_rally_outcome(match_id: str, rally_id: int, req: RallyOutcomeRequest):
    """Tag a rally's outcome. Updates the stored outcomes map."""
    valid = {"p1", "p2", "let", "stroke_p1", "stroke_p2", "warmup"}
    if req.outcome not in valid:
        raise HTTPException(400, f"outcome must be one of {valid}")
    await db.rally_segments.update_one(
        {"match_id": match_id},
        {"$set": {f"outcomes.{rally_id}": req.outcome}},
        upsert=True,
    )
    return {"ok": True, "rally_id": rally_id, "outcome": req.outcome}


@api_router.post("/analysis/rallies/{match_id}/{rally_id}/merge_next")
async def merge_rally_with_next(match_id: str, rally_id: int):
    """Merge rally_id with the next rally (rally_id+1) into one.

    When the ball model loses the ball briefly mid-rally (blur, wall contact),
    it incorrectly splits a real rally into two segments. This endpoint fuses
    them: the first rally absorbs the second's end time and shot count, and the
    second is deleted. A new combined clip is extracted in the background.
    """
    doc = await db.rally_segments.find_one({"match_id": match_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "No segmentation found for this match")

    rallies: list = doc["result"]["rallies"]
    # Merge with the rally that is NEXT BY TIME (not id+1 — ids get non-sequential
    # after edits). Sort by start_t, find this rally, take the one after it.
    order = sorted(range(len(rallies)), key=lambda i: rallies[i]["start_t"])
    pos = next((p for p, i in enumerate(order) if rallies[i]["rally_id"] == rally_id), None)
    if pos is None:
        raise HTTPException(404, f"Rally {rally_id} not found")
    if pos + 1 >= len(order):
        raise HTTPException(404, "No next rally to merge with")
    a = rallies[order[pos]]
    b = rallies[order[pos + 1]]
    absorbed_id = b["rally_id"]

    # Merge b into a
    a["end_t"] = b["end_t"]
    a["duration_s"] = round(b["end_t"] - a["start_t"], 2)
    a["shots"] = a.get("shots", 0) + b.get("shots", 0)
    a["ball_samples"] = a.get("ball_samples", 0) + b.get("ball_samples", 0)
    a["strike_times"] = (a.get("strike_times", []) or []) + (b.get("strike_times", []) or [])

    rallies_new = [r for r in rallies if r["rally_id"] != absorbed_id]
    doc["result"]["rallies"] = rallies_new

    await db.rally_segments.update_one(
        {"match_id": match_id},
        {"$set": {"result.rallies": rallies_new, "result.user_corrected": True},
         "$unset": {f"outcomes.{absorbed_id}": ""}},
    )
    if match_id in RALLY_STATE:
        RALLY_STATE[match_id]["rallies"] = rallies_new

    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if match and extract_rally_clip:
        video_path = str(UPLOAD_DIR / match["video_filename"])
        clip_path = str(RALLY_CLIPS_DIR / match_id / f"rally_{a['rally_id']}.mp4")
        asyncio.create_task(asyncio.to_thread(
            extract_rally_clip, video_path, a["start_t"], a["end_t"], clip_path))
        old_clip = RALLY_CLIPS_DIR / match_id / f"rally_{absorbed_id}.mp4"
        if old_clip.exists():
            old_clip.unlink()

    return {"ok": True, "merged_into": a["rally_id"], "absorbed": absorbed_id,
            "new_end_t": a["end_t"], "new_duration_s": a["duration_s"]}


@api_router.post("/analysis/rallies/{match_id}/{rally_id}/split")
async def split_rally(match_id: str, rally_id: int, at_t: Optional[float] = None):
    """Split a rally into two. By default splits at the rally's largest internal
    quiet (the most likely missed boundary); pass ?at_t= to split at a chosen time.
    The new half gets a fresh rally_id so the clip mapping stays stable."""
    doc = await db.rally_segments.find_one({"match_id": match_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "No segmentation found for this match")
    rallies: list = doc["result"]["rallies"]
    idx = next((i for i, r in enumerate(rallies) if r["rally_id"] == rally_id), None)
    if idx is None:
        raise HTTPException(404, f"Rally {rally_id} not found")
    a = rallies[idx]

    # Decide the split time.
    split_t = at_t
    strikes = a.get("strike_times") or []
    if split_t is None:
        if len(strikes) >= 2:
            gaps = [(strikes[i + 1] - strikes[i], (strikes[i + 1] + strikes[i]) / 2)
                    for i in range(len(strikes) - 1)]
            split_t = max(gaps, key=lambda g: g[0])[1]   # midpoint of biggest gap
        else:
            split_t = round((a["start_t"] + a["end_t"]) / 2, 2)
    if not (a["start_t"] < split_t < a["end_t"]):
        raise HTTPException(400, "split time must be inside the rally")
    split_t = round(split_t, 2)

    new_id = max(r["rally_id"] for r in rallies) + 1
    s_first = [t for t in strikes if t <= split_t]
    s_second = [t for t in strikes if t > split_t]
    first = {**a, "end_t": split_t, "duration_s": round(split_t - a["start_t"], 2),
             "shots": len(s_first) if strikes else a.get("shots", 0) // 2,
             "strike_times": s_first}
    second = {**a, "rally_id": new_id, "start_t": split_t,
              "duration_s": round(a["end_t"] - split_t, 2),
              "shots": len(s_second) if strikes else a.get("shots", 0) - a.get("shots", 0) // 2,
              "strike_times": s_second}
    rallies[idx] = first
    rallies.insert(idx + 1, second)
    doc["result"]["rallies"] = rallies

    await db.rally_segments.update_one(
        {"match_id": match_id},
        {"$set": {"result.rallies": rallies, "result.user_corrected": True}})
    if match_id in RALLY_STATE:
        RALLY_STATE[match_id]["rallies"] = rallies

    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if match and extract_rally_clip:
        video_path = str(UPLOAD_DIR / match["video_filename"])
        for rr in (first, second):
            out = str(RALLY_CLIPS_DIR / match_id / f"rally_{rr['rally_id']}.mp4")
            asyncio.create_task(asyncio.to_thread(
                extract_rally_clip, video_path, rr["start_t"], rr["end_t"], out))

    return {"ok": True, "split_at": split_t, "first_id": rally_id, "new_id": new_id}


@api_router.post("/analysis/rallies/{match_id}/confirm-boundaries")
async def confirm_boundaries(match_id: str):
    """Mark the current rally boundaries as human-confirmed ground truth — the
    label set used to tune the serve-aware auto-segmenter."""
    doc = await db.rally_segments.find_one({"match_id": match_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "No segmentation for this match")
    rallies = doc["result"]["rallies"]
    gt = [{"start_t": r["start_t"], "end_t": r["end_t"]} for r in rallies]
    await db.rally_boundary_labels.replace_one(
        {"match_id": match_id},
        {"match_id": match_id, "boundaries": gt, "num_rallies": len(gt),
         "confirmed_at": datetime.now(timezone.utc).isoformat()},
        upsert=True,
    )
    await db.rally_segments.update_one(
        {"match_id": match_id}, {"$set": {"result.boundaries_confirmed": True}})
    return {"ok": True, "labeled_rallies": len(gt)}


# ===================== BALL TRACE (annotated video) =====================
TRACES_DIR = ROOT_DIR / "traces"
TRACES_DIR.mkdir(exist_ok=True)
TRACE_STATE: Dict[str, Any] = {}   # match_id -> {status, ...}


async def _run_trace(match_id: str, video_path: str, start_s: float,
                     duration_s: float, setup: str, smooth: bool):
    out_path = str(TRACES_DIR / f"{match_id}.mp4")
    TRACE_STATE[match_id] = {"status": "running"}
    try:
        report = await asyncio.to_thread(
            trace_ball_video, video_path, out_path, start_s, duration_s, setup, smooth
        )
        report["status"] = "done"
        TRACE_STATE[match_id] = report
        logger.info(f"Ball trace done for {match_id}: {report.get('detections')} dets")
    except Exception as e:
        logger.error(f"Ball trace failed for {match_id}: {e}")
        TRACE_STATE[match_id] = {"status": "failed", "error": str(e)}


class TraceRequest(BaseModel):
    start_s: float = 30.0
    duration_s: float = 8.0
    smooth: bool = True


@api_router.post("/analysis/trace/{match_id}")
async def generate_trace(match_id: str, req: TraceRequest):
    """Generate a ball-trace video (comet trail) for a window (runs in background)."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if TRACE_STATE.get(match_id, {}).get("status") == "running":
        return {"message": "Trace already generating", "state": TRACE_STATE[match_id]}
    video_path = str(UPLOAD_DIR / match["video_filename"])
    asyncio.create_task(
        _run_trace(match_id, video_path, req.start_s, req.duration_s,
                   _match_setup(match), req.smooth)
    )
    return {"message": "Trace generating", "match_id": match_id}


@api_router.get("/analysis/trace/{match_id}/status")
async def trace_status(match_id: str):
    return TRACE_STATE.get(match_id, {"status": "idle"})


@api_router.get("/analysis/trace/{match_id}/video")
async def trace_video(match_id: str):
    """Serve the generated trace mp4."""
    from fastapi.responses import FileResponse
    path = TRACES_DIR / f"{match_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No trace generated yet")
    return FileResponse(str(path), media_type="video/mp4",
                        headers={"Cache-Control": "no-cache"})


# ===================== SELF-TRAINING (ACTIVE LEARNING) =====================
# The trained model proposes physics-valid ball arcs; the human approves/rejects
# in bulk; approvals become ball labels, rejections become hard negatives. This
# scales the dataset without frame-by-frame labelling.

class MineRequest(BaseModel):
    start_s: float = 0.0
    duration_s: float = 20.0
    min_quality: float = 80.0


@api_router.post("/selftrain/mine/{match_id}")
async def selftrain_mine(match_id: str, req: MineRequest):
    """Run the trained model over a window and store physics-valid proposals."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    video_path = str(UPLOAD_DIR / match["video_filename"])

    result = await asyncio.to_thread(
        mine_ball_tracks, video_path, req.start_s, req.duration_s, req.min_quality,
        12, _match_setup(match),
    )

    # Persist each proposal for review.
    stored = []
    for p in result.get("proposals", []):
        doc = {
            "id": str(uuid.uuid4()),
            "match_id": match_id,
            "detector": result.get("detector"),
            "quality": p["quality"],
            "mean_confidence": p["mean_confidence"],
            "num_points": p["num_points"],
            "overlay_b64": p.get("overlay_b64"),
            "points": p["points"],
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await db.ball_proposals.insert_one(doc.copy())
        doc.pop("_id", None)
        stored.append(doc)
    return {
        "match_id": match_id, "detector": result.get("detector"),
        "num_proposals": len(stored), "proposals": stored,
    }


@api_router.get("/selftrain/proposals/{match_id}")
async def selftrain_proposals(match_id: str):
    """Pending model proposals awaiting human review."""
    props = await db.ball_proposals.find(
        {"match_id": match_id, "status": "pending"}, {"_id": 0}
    ).sort("quality", -1).to_list(100)
    return {"match_id": match_id, "proposals": props}


class ReviewRequest(BaseModel):
    proposal_id: str
    decision: str  # "approve" | "reject"


@api_router.post("/selftrain/review/{match_id}")
async def selftrain_review(match_id: str, req: ReviewRequest):
    """Approve (→ ball labels) or reject (→ hard negative) a proposed track."""
    if req.decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="decision must be approve|reject")
    prop = await db.ball_proposals.find_one(
        {"id": req.proposal_id, "match_id": match_id}, {"_id": 0}
    )
    if not prop:
        raise HTTPException(status_code=404, detail="Proposal not found")

    label = "ball" if req.decision == "approve" else "not_ball"
    points = (
        [{"frame_index": p["frame_index"], "timestamp": p.get("timestamp"),
          "x": p["x"], "y": p["y"]} for p in prop["points"]]
        if label == "ball" else []
    )
    await db.ball_labels.insert_one({
        "id": str(uuid.uuid4()),
        "match_id": match_id,
        "task_id": "selftrain",
        "track_id": -1,
        "label": label,
        "source": "selftrain",
        "points": points,
        "num_points": len(points),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    await db.ball_proposals.update_one(
        {"id": req.proposal_id}, {"$set": {"status": req.decision + "d"}}
    )
    return {"message": f"Proposal {req.decision}d",
            "label": label, "ball_points_added": len(points)}


# ----- Batch mining across the whole library (background) -----
MINE_ALL_STATE: Dict[str, Any] = {"status": "idle"}


async def _run_mine_all(videos: List[Dict], start_s: float, duration_s: float,
                        min_quality: float):
    MINE_ALL_STATE.clear()
    MINE_ALL_STATE.update({"status": "running", "processed": 0,
                           "total": len(videos), "proposals": 0})
    try:
        total_props = 0
        for i, v in enumerate(videos):
            path = str(UPLOAD_DIR / v["video_filename"])
            try:
                # Scan several windows per video so we actually catch rallies.
                result = await asyncio.to_thread(
                    scan_video_for_arcs, path, 4, duration_s, min_quality, 3,
                    _match_setup(v),
                )
                for p in result.get("proposals", []):
                    await db.ball_proposals.insert_one({
                        "id": str(uuid.uuid4()), "match_id": v["id"],
                        "detector": result.get("detector"),
                        "quality": p["quality"], "mean_confidence": p["mean_confidence"],
                        "num_points": p["num_points"], "overlay_b64": p.get("overlay_b64"),
                        "points": p["points"], "status": "pending",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    })
                    total_props += 1
            except Exception as e:
                logger.error(f"mine-all failed for {v['id']}: {e}")
            MINE_ALL_STATE.update({"processed": i + 1, "proposals": total_props})
        MINE_ALL_STATE.update({"status": "done", "proposals": total_props})
    except Exception as e:
        MINE_ALL_STATE.clear()
        MINE_ALL_STATE.update({"status": "failed", "error": str(e)})


class MineAllRequest(BaseModel):
    start_s: float = 30.0
    duration_s: float = 12.0
    min_quality: float = 80.0
    max_videos: int = 40


@api_router.post("/selftrain/mine-all")
async def selftrain_mine_all(req: MineAllRequest):
    """Mine ball-arc proposals across many library videos in the background."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Perception spine unavailable")
    if MINE_ALL_STATE.get("status") == "running":
        return {"message": "Batch mining already running", "state": MINE_ALL_STATE}
    if not os.path.exists(BALL_WEIGHTS_PATH):
        raise HTTPException(status_code=400,
                            detail="No trained model — train one before mining.")

    videos = await db.matches.find(
        {"video_filename": {"$exists": True}},
        {"_id": 0, "id": 1, "video_filename": 1, "source": 1},
    ).sort("upload_time", -1).to_list(req.max_videos)

    asyncio.create_task(
        _run_mine_all(videos, req.start_s, req.duration_s, req.min_quality)
    )
    return {"message": "Batch mining started", "videos": len(videos)}


@api_router.get("/selftrain/mine-all/status")
async def selftrain_mine_all_status():
    return MINE_ALL_STATE


@api_router.get("/selftrain/proposals")
async def selftrain_all_proposals(limit: int = 60):
    """All pending proposals across videos (for one-pass review), top by quality."""
    props = await db.ball_proposals.find(
        {"status": "pending"}, {"_id": 0}
    ).sort("quality", -1).to_list(limit)
    # attach match titles
    titles = {}
    for p in props:
        mid = p["match_id"]
        if mid not in titles:
            m = await db.matches.find_one({"id": mid}, {"_id": 0, "title": 1})
            titles[mid] = m.get("title", "?") if m else "?"
        p["match_title"] = titles[mid]
    return {"proposals": props, "count": len(props)}


@api_router.get("/selftrain/stats")
async def selftrain_stats():
    """Self-training progress across all videos."""
    by_status = {}
    async for d in db.ball_proposals.aggregate([
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]):
        by_status[d["_id"]] = d["count"]
    selftrain_points = await db.ball_labels.aggregate([
        {"$match": {"source": "selftrain", "label": "ball"}},
        {"$group": {"_id": None, "pts": {"$sum": "$num_points"}}},
    ]).to_list(1)
    return {
        "proposals_by_status": by_status,
        "selftrain_ball_points": selftrain_points[0]["pts"] if selftrain_points else 0,
    }


# ===================== STRUCTURED TIMELINE (M2) =====================
# Build a structured rally timeline (ball trajectory -> shot-contact events ->
# rallies) for a window. Uses the trained TrackNet ball detector when weights
# exist, else the classical detector (so events are rough until the model trains).

class TimelineRequest(BaseModel):
    start_s: float = 0.0
    duration_s: float = 8.0


@api_router.post("/analysis/timeline/{match_id}")
async def build_match_timeline(match_id: str, req: TimelineRequest):
    """Build & store a structured rally timeline for a window. Needs court calibration."""
    if not PERCEPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Perception spine unavailable")
    match = await db.matches.find_one({"id": match_id}, {"_id": 0})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    calib = match.get("court_calibration")
    if not calib:
        raise HTTPException(
            status_code=400,
            detail="Court not calibrated. Mark the court first (movement + timeline need it).",
        )

    video_path = str(UPLOAD_DIR / match["video_filename"])
    cap = cv2.VideoCapture(video_path)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
    cap.release()
    calibration = CourtCalibration.from_normalized(calib, vw, vh)

    timeline = await asyncio.to_thread(
        analyze_rally_window, video_path, calibration, req.start_s, req.duration_s,
        "yolo11n.pt", None, _match_setup(match),
    )

    doc = {
        "id": str(uuid.uuid4()),
        "match_id": match_id,
        "start_s": req.start_s,
        "duration_s": req.duration_s,
        "timeline": timeline,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.timelines.insert_one(doc.copy())
    doc.pop("_id", None)
    return doc


@api_router.get("/analysis/timeline/{match_id}")
async def get_match_timelines(match_id: str):
    """Return stored structured timelines for a match."""
    docs = await db.timelines.find(
        {"match_id": match_id}, {"_id": 0}
    ).sort("created_at", -1).to_list(50)
    return {"match_id": match_id, "timelines": docs}


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
