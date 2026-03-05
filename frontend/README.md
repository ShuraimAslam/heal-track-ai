# Heal-Track Fancy Web UI

This folder contains the modern web interface for the Heal-Track AI project.

## Project Structure

- `api.py`: FastAPI bridge that wraps the existing ML logic in `../src`.
- `web/`: Next.js frontend application with premium UI and animations.
- `streamlit_app.py`: Legacy Streamlit application (preserved for reference).

## How to Run

### 1. Start the API Server
Ensure you have the required Python packages installed:
```bash
pip install fastapi uvicorn python-multipart Pillow
```
Then run the API:
```bash
python api.py
```
The API will be available at `http://localhost:8000`.

### 2. Start the Web UI
Navigate to the `web` directory:
```bash
cd web
npm run dev
```
The fancy web interface will be available at `http://localhost:3000`.

## Features
- **Cinematic Entrance**: Smooth animations using Framer Motion.
- **Glassmorphism Design**: Modern, premium aesthetic.
- **AI Dashboard**: Interactive wound analysis with metrics and reports.
- **Real-time Visualization**: High-fidelity overlay of segmentation results.
