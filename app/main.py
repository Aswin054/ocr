import base64  # Add this with your other imports
import asyncio
import platform
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import logging
from pathlib import Path

from app.detector import LicensePlateDetector
from app.config import settings

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="License Plate Detection API",
    description="API for detecting and recognizing license plates in images",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize detector
try:
    detector = LicensePlateDetector(settings.MODEL_PATH)
    logger.info("License plate detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize detector: {str(e)}")
    raise RuntimeError("Could not initialize detector") from e

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the frontend interface"""
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load frontend: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load frontend")

@app.post("/api/detect")
async def detect_license_plate(image: UploadFile = File(...)):
    """
    Detect and recognize license plates in an uploaded image
    
    Returns:
    - plates: List of detected license plates with text and confidence
    - annotated_image: Base64 encoded image with bounding boxes
    """
    try:
        # Validate input
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read and decode image
        contents = await image.read()
        np_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=400, 
                detail="Could not decode image. Please upload a valid image file."
            )
        
        # Process image
        plates, annotated_img = detector.detect_and_recognize(img)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse(content={
            "success": True,
            "plates": plates,
            "annotated_image": encoded_img
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing the image"
        ) from e

def start():
    """Start the application with Uvicorn"""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )

if __name__ == "__main__":
    start()