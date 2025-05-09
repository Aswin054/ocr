import cv2
import easyocr
import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

# Initialize logger
logger = logging.getLogger(__name__)

class LicensePlateDetector:
    def __init__(self, model_path: str):
        """
        Initialize the license plate detector with YOLOv5 model.
        
        Args:
            model_path: Path to the YOLOv5 model weights file
        """
        self.reader = None
        self.model = None
        self._initialize_components(model_path)
        
    def _initialize_components(self, model_path: str):
        """Initialize OCR reader and detection model"""
        try:
            # Initialize EasyOCR first (can take some time)
            logger.info("Initializing EasyOCR...")
            self.reader = easyocr.Reader(['en'])
            
            # Then load the detection model
            logger.info("Loading YOLOv5 model...")
            self.model = self._load_model(model_path)
            
            logger.info("License plate detector initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError("Could not initialize detector components") from e
    
    def _load_model(self, model_path: str):
        """
        Load YOLOv5 model with fallback for different versions.
        
        Args:
            model_path: Path to the model weights file
            
        Returns:
            Loaded YOLOv5 model
        """
        try:
            # Try loading with ultralytics YOLO (newer versions)
            from ultralytics import YOLO
            model = YOLO(model_path)
            logger.info("Loaded model using Ultralytics YOLO")
            return model
        except ImportError:
            try:
                # Fallback to torch hub loading
                model = torch.hub.load(
                    'ultralytics/yolov5:v7.0', 
                    'custom',
                    path=model_path,
                    force_reload=True,
                    trust_repo=True
                )
                logger.info("Loaded model using torch hub")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError(f"Could not load model from {model_path}") from e
    
    def preprocess_plate(self, plate_region: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate region for better OCR results.
        
        Args:
            plate_region: Cropped license plate image
            
        Returns:
            Preprocessed grayscale image
        """
        try:
            plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            plate_enhanced = cv2.equalizeHist(plate_gray)
            return plate_enhanced
        except Exception as e:
            logger.error(f"Error in plate preprocessing: {str(e)}")
            return plate_region  # Return original if processing fails
    
    def detect_and_recognize(self, image_array: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect license plates in an image and recognize text.
        
        Args:
            image_array: Input image as numpy array
            
        Returns:
            Tuple of (detected_plates, annotated_image)
        """
        if not isinstance(image_array, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        try:
            img = image_array.copy()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.model(img_rgb)
            plates = []
            
            # Handle different YOLO versions
            if hasattr(results, 'pandas'):  # Older YOLOv5
                detections = results.pandas().xyxy[0]
                for _, det in detections.iterrows():
                    if det['confidence'] < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                    plate_region = img[y1:y2, x1:x2]
                    
                    # Preprocess and OCR
                    plate_processed = self.preprocess_plate(plate_region)
                    ocr_results = self.reader.readtext(plate_processed)
                    
                    if ocr_results:
                        plates.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(det['confidence']),
                            'text': ocr_results[0][1],
                            'ocr_confidence': float(ocr_results[0][2])
                        })
            else:  # Newer YOLO versions
                for det in results[0].boxes:
                    if det.conf < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    plate_region = img[y1:y2, x1:x2]
                    
                    # Preprocess and OCR
                    plate_processed = self.preprocess_plate(plate_region)
                    ocr_results = self.reader.readtext(plate_processed)
                    
                    if ocr_results:
                        plates.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(det.conf),
                            'text': ocr_results[0][1],
                            'ocr_confidence': float(ocr_results[0][2])
                        })
            
            # Annotate image
            annotated_img = img.copy()
            for plate in plates:
                x1, y1, x2, y2 = plate['bbox']
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_img, 
                    f"{plate['text']} ({plate['confidence']:.2f})", 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )
            
            return plates, annotated_img
            
        except Exception as e:
            logger.error(f"Error in detection/recognition: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to process image") from e