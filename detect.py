import cv2
import easyocr
import torch
import numpy as np

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def load_yolov5_model(model_path):
    try:
        # Try loading with ultralytics YOLO (newer versions)
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model
    except ImportError:
        # Fallback to torch hub loading
        model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', 
                             path=model_path,
                             force_reload=True,
                             trust_repo=True)
        return model

# Load your custom model
model = load_yolov5_model(r'C:\Users\Lenova\Desktop\ocr\models\license_plate_detector.pt')

def detect_and_recognize_license_plate(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform detection
    results = model(img_rgb)
    
    # Process detections
    plates = []
    
    # Handle different YOLO versions
    if hasattr(results, 'pandas'):  # Older YOLOv5
        detections = results.pandas().xyxy[0]
        for _, det in detections.iterrows():
            if det['confidence'] < 0.5:
                continue
            x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
            plate_region = img[y1:y2, x1:x2]
            
            # Process and OCR within this block
            plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            plate_enhanced = cv2.equalizeHist(plate_gray)
            ocr_results = reader.readtext(plate_enhanced)
            
            if ocr_results:
                plates.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(det['confidence']),
                    'text': ocr_results[0][1],
                    'ocr_confidence': ocr_results[0][2]
                })
    else:  # Newer YOLO versions
        for det in results[0].boxes:
            if det.conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            plate_region = img[y1:y2, x1:x2]
            
            # Process and OCR within this block
            plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            plate_enhanced = cv2.equalizeHist(plate_gray)
            ocr_results = reader.readtext(plate_enhanced)
            
            if ocr_results:
                plates.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(det.conf),
                    'text': ocr_results[0][1],
                    'ocr_confidence': ocr_results[0][2]
                })
    
    # Draw results
    output_img = img.copy()
    for plate in plates:
        x1, y1, x2, y2 = plate['bbox']
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_img, f"{plate['text']} ({plate['confidence']:.2f})", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('License Plate Detection', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return plates

if __name__ == "__main__":
    image_path = r'C:\Users\Lenova\Desktop\ocr\images\test2.jpg'
    results = detect_and_recognize_license_plate(image_path)
    
    if results:
        print("\nLicense Plate Detection Results:")
        for i, plate in enumerate(results, 1):
            print(f"Plate {i}:")
            print(f"  Bounding Box: {plate['bbox']}")
            print(f"  Detection Confidence: {plate['confidence']:.2f}")
            print(f"  Recognized Text: {plate['text']}")
            print(f"  OCR Confidence: {plate['ocr_confidence']:.2f}")
    else:
        print("No license plates detected.")