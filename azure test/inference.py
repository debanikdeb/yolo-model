from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import json

# Function to load the YOLO model


def init():
    global model
    model = YOLO("yolo-v11-s-no-aug.pt")  # Replace with your actual model path


# Function to run inference on an image


def run(input_data):
    # Get image URL from input
    img_url = input_data["image_url"]
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))

    # Run the YOLO model
    results = model.predict(source=img)

    # Format the results as a JSON response
    json_result = []
    for result in results:
        for box in result.boxes:
            # Extract the xywh values from the first (and only) row
            xywh = box.xywh[0]
            json_result.append(
                {
                    "class": int(box.cls.item()),
                    "x_center": xywh[0].item(),
                    "y_center": xywh[1].item(),
                    "width": xywh[2].item(),
                    "height": xywh[3].item(),
                }
            )

    return json.dumps(json_result)

    # return json.dumps(json_result)
