# SquashSense AI - Product Requirements Document

> **See [ML_ROADMAP.md](ML_ROADMAP.md)** for the technical plan to a squash-specialized
> model (perception spine → labeled dataset → trained shot classifier → LLM reasoning).
> The original GPT-vision frame-sampling approach below is superseded for fine-grained
> shot analysis; movement metrics are now produced by the perception spine without an LLM.

## Original Problem Statement
Build the world's first AI-powered Squash analysis software that can take any squash match video and analyze it to give valuable information including shot categorization, rallies, swings, player movement, court coverage, and more.

## User Personas
1. **Amateur Squash Players** - Want to improve their game by understanding patterns
2. **Professional Players** - Need detailed analytics for competition preparation
3. **Squash Coaches** - Analyze student performance and identify areas for improvement
4. **Sports Analysts** - Generate reports and insights for clubs/teams

## Core Requirements (Static)
- Video upload (MP4, MOV, AVI, WebM)
- AI-powered frame analysis using GPT-5.2 vision
- Shot categorization (drive, drop, boast, volley, lob, kill, serve)
- Rally analysis with length, patterns, and winning shots
- Player movement tracking and court coverage
- Swing mechanics analysis (forehand/backhand ratios)
- Match history storage
- Export reports (PDF/JSON)

## What's Been Implemented (MVP - January 2026)

### Backend (FastAPI + MongoDB)
- [x] Video upload endpoint with file validation
- [x] Frame extraction using OpenCV
- [x] AI analysis using GPT-5.2 vision via emergentintegrations
- [x] Background processing for video analysis
- [x] Match CRUD operations (create, read, delete)
- [x] Export endpoints (JSON, PDF)
- [x] Health check endpoint

### Frontend (React + Tailwind + Shadcn UI)
- [x] Landing page with hero section and features
- [x] Upload page with drag-and-drop
- [x] History page with match list and actions
- [x] Analysis dashboard with 5 tabs:
  - Shot distribution (pie chart + breakdown)
  - Rally analysis (bar chart + timeline)
  - Player comparison stats
  - Court movement heatmap
  - Key insights
- [x] Export functionality (PDF/JSON)
- [x] Modern dark theme with volt green accents

### Design
- "The Kinetic Analyst" theme
- Barlow Condensed + Inter + JetBrains Mono fonts
- Dark mode (#050505 background)
- Volt Green (#DFFF00) primary accent
- Electric Cyan (#00F0FF) secondary accent
- Bento grid dashboard layout

## Prioritized Backlog

### P0 (Critical)
- All core features implemented in MVP ✅

### P1 (High Priority - Next Phase)
- [ ] Real-time progress updates via WebSocket
- [ ] Video playback with timestamp sync to shots
- [ ] Improved AI accuracy with more frames
- [ ] User authentication system
- [ ] Match comparison view (side-by-side analysis)

### P2 (Medium Priority)
- [ ] Court coverage heatmap with actual heat visualization
- [ ] Shot trajectory overlay on video
- [ ] Training recommendations based on analysis
- [ ] Share analysis via link
- [ ] Mobile responsive improvements

### P3 (Nice to Have)
- [ ] Multi-match tournament analysis
- [ ] Player profile management
- [ ] Historical trend charts
- [ ] Integration with fitness trackers
- [ ] Slow-motion playback controls

## Next Tasks List
1. Add WebSocket for real-time analysis progress
2. Implement video playback synced with shot timestamps
3. Add user authentication (JWT or Google OAuth)
4. Create match comparison feature
5. Improve AI prompt for better shot detection accuracy

## Technical Architecture
```
Frontend (React 19)          Backend (FastAPI)           Database (MongoDB)
     |                              |                          |
     |------ REST API ------------>|                          |
     |                              |------ CRUD ------------->|
     |<----- JSON Response --------|                          |
     |                              |                          |
     |                    [OpenCV Frame Extraction]            |
     |                              |                          |
     |                    [GPT-5.2 Vision Analysis]            |
     |                              |                          |
     |                    [Background Processing]              |
```
