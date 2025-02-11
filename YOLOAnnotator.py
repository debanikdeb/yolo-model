import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
import tempfile
import matplotlib.pyplot as plt
from typing import List, Tuple
from io import BytesIO
from PIL import Image
from fastapi.responses import StreamingResponse

# Create the FastAPI app instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domains if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

model_path = 'yolo-v11.pt'

# Define the YOLOAnnotator class
class YOLOAnnotator:
    def __init__(self, model_path=model_path):
        # Define colors for the five classes using hex codes
        self.colors = {
            0: self.hex_to_bgr('#000000'),  # Black
            1: self.hex_to_bgr('#76B93E'),  # Green (#76B93E)
            2: self.hex_to_bgr('#FFBD59'),  # Yellow (#FFBD59)
            3: self.hex_to_bgr('#FF1616'),  # Red (#FF1616)
            4: self.hex_to_bgr('#00FFFF')   # Cyan (#00FFFF)
        }

        # Load the model
        self.model = YOLO(model_path)
        print("Model Loaded")

    def hex_to_bgr(self, hex_color):
        """Convert hex color code to BGR tuple."""
        hex_color = hex_color.lstrip('#')
        return (int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16))

    def resize_image(self, image: BytesIO, size=(640, 640)):
        """Read and resize the image."""
        img = Image.open(image)
        img = img.convert('RGB')  # Ensure image is in RGB format
        img = img.resize(size)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def extract_and_annotate_boxes(self, image: BytesIO):
        """Extract bounding boxes, annotate the image, and return results."""
        # Resize the image to match model input size
        resized_img = self.resize_image(image, size=(640, 640))

        # Perform inference on the resized image
        results = self.model([resized_img])  # Pass the resized image directly

        # Initialize bounding box count and list for coordinates
        total_boxes = 0
        coordinates = []

        # Extract bounding boxes and annotate the image
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                total_boxes += len(boxes.xyxy)  # Count the number of bounding boxes
                for i in range(len(boxes.xyxy)):
                    # Extract coordinates and convert to list
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                    class_id = int(boxes.cls[i].item())

                    # Draw bounding box with the specified color
                    color = self.colors.get(class_id, (0, 255, 0))  # Default to green if class_id not found
                    cv2.rectangle(resized_img, (x1, y1), (x2, y2), color, 2)

                    # Draw label with class ID only
                    label = f"{class_id}"
                    cv2.putText(resized_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Print coordinates of the bounding box
                    print(f"Bounding Box {i+1}: ({x1}, {y1}), ({x2}, {y2})")
                    coordinates.append(((x1, y1), (x2, y2)))
            else:
                print("No bounding boxes detected.")

        # Print total number of bounding boxes
        print(f"Total Bounding Boxes: {total_boxes}")

        # Save the annotated image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            annotated_img_path = temp_file.name
            # Save the annotated image to the temporary file
            cv2.imwrite(annotated_img_path, resized_img)
            print(f"Annotated image saved temporarily in {annotated_img_path}")

        # Return the path to the annotated image and the coordinates
        return annotated_img_path, coordinates

# Instantiate the YOLOAnnotator
yolo_annotator = YOLOAnnotator(model_path=model_path)

# Define an endpoint to upload images and get annotations
@app.post("/annotate/")
async def annotate_image(file: UploadFile = File(...)):
    # Perform annotation
    annotated_img_path, coordinates = yolo_annotator.extract_and_annotate_boxes(file.file)

    # Open the annotated image file for reading as binary
    annotated_img_file = open(annotated_img_path, "rb")

    # Return the annotated image as a StreamingResponse
    return StreamingResponse(annotated_img_file, media_type="image/jpeg")
