# License Plate Detection System (Render Deployment)

A FastAPI-based application for detecting and recognizing license plates in images, optimized for Render.com deployment.

## Features

- License plate detection using YOLOv5
- Text recognition using EasyOCR
- Web interface for easy testing
- REST API for integration
- Optimized for Render.com deployment

## Deployment on Render

1. Create a new **Web Service** on Render
2. Connect your GitHub repository
3. Set the following configuration:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000`
4. Add environment variables:
   - `MODEL_PATH`: `models/license_plate_detector.pt`
   - `PYTHON_VERSION`: `3.9.16`
5. Deploy!

## Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your YOLOv5 model in `models/license_plate_detector.pt`
4. Run: `uvicorn app.main:app --reload`

## API Endpoints

- `GET /`: Web interface
- `POST /api/detect`: Process an image and return detection results

## Environment Variables

- `MODEL_PATH`: Path to YOLOv5 model (default: `models/license_plate_detector.pt`)
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `DEBUG`: Enable debug mode (default: `false`)
- `WORKERS`: Number of worker processes (default: `1`)