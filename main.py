import cv2
import numpy as np
from ultralytics import YOLO

class WebcamTracker:
    def __init__(self, model_name="yolov8n.pt", process_every_n_frames=2):
        """
        Initialize the webcam tracker with YOLOv8 (free, local).
        
        Args:
            model_name: YOLOv8 model to use ('yolov8n.pt' is nano - fastest)
            process_every_n_frames: Process every nth frame to improve performance
        """
        # Load YOLOv8 model (automatically downloads on first run)
        self.model = YOLO(model_name)
        self.process_every_n_frames = process_every_n_frames
        self.frame_count = 0
        self.last_detections = []
        
    def identify_objects(self, frame):
        """
        Use YOLOv8 to identify objects in the frame.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            tuple: (results object, detection string)
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Extract detected objects
        detections = results[0]
        class_names = detections.names
        
        detected_objects = []
        for box in detections.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = class_names[class_id]
            
            if confidence > 0.5:  # Only show detections with >50% confidence
                detected_objects.append(f"{object_name} ({confidence:.1%})")
        
        # Create detection string
        if detected_objects:
            detection_text = " | ".join(detected_objects[:5])  # Limit to 5 objects
        else:
            detection_text = "No objects detected"
        
        return results[0], detection_text
    
    def run(self):
        """
        Start the live webcam tracking.
        Press 'q' to quit.
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam opened. Press 'q' to quit.")
        print("=" * 60)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Display the frame
            display_frame = frame.copy()
            
            # Process every nth frame for performance
            if self.frame_count % self.process_every_n_frames == 0:
                try:
                    # Identify objects in the current frame using YOLO
                    results, detection_text = self.identify_objects(frame)
                    self.last_detections = detection_text
                    
                    # Draw bounding boxes on frame
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        object_name = results.names[class_id]
                        
                        if confidence > 0.5:
                            # Draw rectangle
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Draw label
                            label = f"{object_name} {confidence:.1%}"
                            cv2.putText(display_frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Print to console
                    print(f"Frame {self.frame_count}: {detection_text}")
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    error_text = "Detection Error"
                    cv2.putText(display_frame, error_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display last detection results
            if self.last_detections:
                # Wrap text if too long
                words = self.last_detections.split("|")
                for i, word in enumerate(words[:3]):
                    y_pos = 30 + (i * 25)
                    cv2.putText(display_frame, word.strip(), (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Add frame counter
            cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, display_frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Resize for display if too large
            display_frame = cv2.resize(display_frame, (1024, 768))
            cv2.imshow('Govt_404 - Live Webcam Tracker', display_frame)
            
            self.frame_count += 1
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nShutting down...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Tracker stopped.")


if __name__ == "__main__":
    tracker = WebcamTracker(process_every_n_frames=5)
    tracker.run()
