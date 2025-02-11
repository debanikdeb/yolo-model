import os
import logging
import json
import tempfile
from io import BytesIO
import base64
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Global variable for the model and YOLOAnnotator
model = None
yolo_annotator = None

def init():
    """
    This function is called when the container is initialized/started,
    typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model, yolo_annotator
    logging.info("Init started")

    # Load the model
    model_path = os.path.join("yolo-v11.pt")
    try:
        model = YOLO(model_path)
        logging.info("Model loaded from %s", model_path)
        # Instantiate the YOLOAnnotator after the model is loaded
        yolo_annotator = YOLOAnnotator()
        logging.info("YOLOAnnotator instantiated")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        model = None

    logging.info("Init completed")

class YOLOAnnotator:
    def __init__(self):
        global model
        logging.info(f"Model in YOLOAnnotator init: {model}")
        self.model = model
        if self.model is None:
            logging.error("Model is None in YOLOAnnotator init")

        # Define colors for the five classes using hex codes
        self.colors = {
            0: self.hex_to_bgr('#000000'),  # Black
            1: self.hex_to_bgr('#76B93E'),  # Green
            2: self.hex_to_bgr('#FFBD59'),  # Yellow
            3: self.hex_to_bgr('#FF1616'),  # Red
            4: self.hex_to_bgr('#00FFFF')   # Cyan
        }

    def hex_to_bgr(self, hex_color):
        """Convert hex color code to BGR tuple."""
        hex_color = hex_color.lstrip('#')
        return (int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16))

    def resize_image(self, image: BytesIO, size=(640, 640)):
        """Read and resize the image."""
        logging.info("Resizing image")
        img = Image.open(image)
        img = img.convert('RGB')  # Ensure image is in RGB format
        img = img.resize(size)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def extract_and_annotate_boxes(self, image: BytesIO):
        """Extract bounding boxes, annotate the image, and return results."""
        # Resize the image to match model input size
        resized_img = self.resize_image(image, size=(640, 640))
        logging.info("Image resized")

        # Perform inference on the resized image
        try:
            results = self.model([resized_img])  # Pass the resized image directly
            logging.info("Model inference completed")
        except Exception as e:
            logging.error(f"Error during model inference: {e}")
            raise

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

                    # Log coordinates of the bounding box
                    logging.info(f"Bounding Box {i+1}: ({x1, y1}), ({x2, y2})")
                    coordinates.append(((x1, y1), (x2, y2)))
            else:
                logging.info("No bounding boxes detected.")

        # Log total number of bounding boxes
        logging.info(f"Total Bounding Boxes: {total_boxes}")

        # Save the annotated image to a temporary file
        _, annotated_img_path = tempfile.mkstemp(suffix='.jpg')
        # Save the annotated image to the temporary file
        cv2.imwrite(annotated_img_path, resized_img)
        logging.info(f"Annotated image saved temporarily in {annotated_img_path}")

        # Return the path to the annotated image and the coordinates
        return annotated_img_path, coordinates

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the YOLO model's inference
    method and return the result back
    """
    logging.info("Run started")

    try:
        # Decode the base64 image data
        logging.info("Decoding image data")
        data = json.loads(raw_data)
        image_data = base64.b64decode(data['file'])
        
        # Convert the decoded data to a file-like object
        file_like = BytesIO(image_data)
        logging.info("File-like object created from image data")
        
        # Perform annotation
        annotated_img_path, coordinates = yolo_annotator.extract_and_annotate_boxes(file_like)
        logging.info("Annotation completed")
        
        # Read the annotated image
        with open(annotated_img_path, "rb") as f:
            annotated_image = f.read()
        logging.info("Annotated image read from file")
        
        # Encode the annotated image to base64
        encoded_annotated_image = base64.b64encode(annotated_image).decode('utf-8')
        logging.info("Annotated image encoded to base64")
        
        # Return the annotated image and coordinates as JSON
        response = {
            "annotated_image": encoded_annotated_image,
            "coordinates": coordinates
        }
        
        logging.info("Run completed")
        logging.info("Predictions: %s", response)
        
        return json.dumps(response)
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    init()
