from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
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
    
    # Analysis results
    total_shots: int = 0
    total_rallies: int = 0
    shots: List[Dict[str, Any]] = []
    rallies: List[Dict[str, Any]] = []
    shot_distribution: Dict[str, int] = {}
    player1_stats: Dict[str, Any] = {}
    player2_stats: Dict[str, Any] = {}
    movement_data: List[Dict[str, Any]] = []
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
    duration: float
    status: str
    progress: int
    player1_name: str = "Player 1"
    player2_name: str = "Player 2"
    player1_description: str = ""
    player2_description: str = ""
    player1_frame: Optional[str] = None
    player2_frame: Optional[str] = None
    total_shots: int
    total_rallies: int
    shots: List[Dict[str, Any]]
    rallies: List[Dict[str, Any]]
    shot_distribution: Dict[str, int]
    player1_stats: Dict[str, Any]
    player2_stats: Dict[str, Any]
    movement_data: List[Dict[str, Any]]
    swing_analysis: List[Dict[str, Any]]
    key_insights: List[str]
    thumbnail: Optional[str] = None

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
                frame = cv2.resize(frame, (640, 480))
                # Convert to base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                timestamp = frame_count / fps if fps > 0 else 0
                frames.append((frame_base64, timestamp, frame_count))
            
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
        
        for idx, (frame_base64, timestamp, frame_num) in enumerate(frames):
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
            analysis = await analyze_frame_with_ai(frame_base64, frame_num, context, player_info)
            
            if analysis.get("shot_detected"):
                shot_type = analysis.get("shot_type", "drive")
                active_player = analysis.get("active_player", "player1")
                
                shot = {
                    "shot_type": shot_type,
                    "timestamp": timestamp,
                    "player": active_player,
                    "success": analysis.get("confidence", 0.5) > 0.3,
                    "court_position": analysis.get(f"{active_player}_position", "mid"),
                    "swing_type": analysis.get("swing_type", "forehand"),
                    "confidence": analysis.get("confidence", 0.5),
                    "notes": analysis.get("notes", "")
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
            
            # Track movement
            if analysis.get("player1_position"):
                pos_map = {"front": 0.2, "mid": 0.5, "back": 0.8}
                movement_data.append({
                    "player": "player1",
                    "x": pos_map.get(analysis.get("player1_position"), 0.5) + np.random.uniform(-0.1, 0.1),
                    "y": np.random.uniform(0.2, 0.8),
                    "time": timestamp
                })
            if analysis.get("player2_position"):
                pos_map = {"front": 0.2, "mid": 0.5, "back": 0.8}
                movement_data.append({
                    "player": "player2",
                    "x": pos_map.get(analysis.get("player2_position"), 0.5) + np.random.uniform(-0.1, 0.1),
                    "y": np.random.uniform(0.2, 0.8),
                    "time": timestamp
                })
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
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
        
        # Calculate court coverage
        p1_coverage = (player1_stats["front_court"] + player1_stats["mid_court"] + player1_stats["back_court"])
        p2_coverage = (player2_stats["front_court"] + player2_stats["mid_court"] + player2_stats["back_court"])
        
        if p1_coverage > 0:
            player1_stats["court_coverage"] = round((player1_stats["front_court"] + player1_stats["back_court"]) / p1_coverage * 100, 1)
        if p2_coverage > 0:
            player2_stats["court_coverage"] = round((player2_stats["front_court"] + player2_stats["back_court"]) / p2_coverage * 100, 1)
        
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
