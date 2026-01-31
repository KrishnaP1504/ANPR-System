import cv2
import easyocr
import time
from ultralytics import YOLO
import re
import os

# --- CONFIGURATION ---
MODEL_PATH = r'C:\Users\pipal\OneDrive\Desktop\Nootbook P\ANPR_System\runs\detect\models\anpr_yolov128\weights\best.pt' 
VIDEO_SOURCE = r"C:\Users\pipal\OneDrive\Desktop\Nootbook P\ANPR_System\video\test\test_video.mp4"
OUTPUT_DIR = r"C:\Users\pipal\OneDrive\Desktop\Nootbook P\ANPR_System\video\Output"
OUTPUT_FILENAME = "result_enhanced.mp4"

CONFIDENCE_THRESHOLD = 0.5

class ANPRSystem:
    def __init__(self, model_path, gpu=False):
        # Initialize YOLO
        print(f"Loading Model from: {model_path}")
        self.detector = YOLO(model_path)
        
        # Initialize EasyOCR
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=gpu) 

    def clean_text(self, text):
        """
        Clean text and fix common OCR confusion errors.
        Example: 'O' -> '0' if it looks like a number.
        """
        # 1. Remove special characters
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # 2. Simple logic correction (Optional but helpful)
        # If the text is exactly 7 chars (Standard format often), we can be stricter.
        # But for now, let's just do general cleaning.
        return text

    def process_video(self, source, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing started... Output: {output_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()

            # 1. Detection
            results = self.detector(frame, stream=True, conf=CONFIDENCE_THRESHOLD, verbose=False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    # 2. Crop the plate
                    plate_crop = frame[y1:y2, x1:x2]

                    # --- NEW: IMAGE ENHANCEMENT STEP ---
                    try:
                        # Convert to grayscale
                        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

                        # A. Upscale (Zoom in) - Makes small text readable
                        # resizing by 3x
                        gray_plate = cv2.resize(gray_plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

                        # B. Contrast/Thresholding - Makes text BLACK and background WHITE (removes gray noise)
                        # Otsu's thresholding automatically finds the best separation
                        gray_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                        
                        # (Optional) You can uncomment this to see what the AI sees:
                        # cv2.imshow("Enhanced Plate", gray_plate)

                        # 3. OCR on the ENHANCED image
                        ocr_result = self.reader.readtext(gray_plate, detail=0)
                        
                        if ocr_result:
                            raw_text = " ".join(ocr_result)
                            clean_plate_text = self.clean_text(raw_text)

                            if len(clean_plate_text) > 3: # Filter out noise like "I" or "1"
                                # Draw
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                (text_w, text_h), _ = cv2.getTextSize(clean_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                                cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_w, y1), (0, 255, 0), -1)
                                cv2.putText(frame, clean_plate_text, (x1, y1 - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                                
                                print(f"Original: {raw_text} -> Clean: {clean_plate_text}")

                    except Exception as e:
                        continue

            fps_calc = 1 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps_calc:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show Result
            cv2.imshow("ANPR System", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(VIDEO_SOURCE):
        # Prepare full output path
        full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        
        app = ANPRSystem(model_path=MODEL_PATH, gpu=True)
        app.process_video(VIDEO_SOURCE, full_output_path)
    else:
        print("Check your paths!")