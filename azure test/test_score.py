from score import run

# Simulate the input data (using an image URL or local path)
input_data = {
    "image_url": "https://dronetjekbackend.blob.core.windows.net/images/original_(2).JPG__7c6de8ab-44b4-4188-90ca-2663b33c0619.JPG"}

# Call the run function as Azure would do
result = run(input_data)

print(result)  # This should return the YOLO model's predictions in JSON format
