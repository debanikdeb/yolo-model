import requests
import base64
import json
import os
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

class ImageAnnotator:
    def __init__(self, connection_string, container_name, scoring_uri, primary_key):
        self.connection_string = connection_string
        self.container_name = container_name
        self.scoring_uri = scoring_uri
        self.primary_key = primary_key
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)

    def generate_sas_token(self, blob_name):
        sas_token = generate_blob_sas(
            account_name=self.blob_service_client.account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            account_key=self.blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        return sas_token

    def upload_image_to_blob(self, image_path):
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            if not container_client.exists():
                container_client.create_container()

            blob_name = os.path.basename(image_path)
            blob_client = container_client.get_blob_client(blob_name)

            with open(image_path, 'rb') as image_file:
                blob_client.upload_blob(image_file, overwrite=True)  # ✅ Overwrite enabled

            sas_token = self.generate_sas_token(blob_name)
            image_url = f"{blob_client.url}?{sas_token}"
            return image_url

        except Exception as e:
            print(f"Error uploading image: {e}")
            return None

    def annotate_image(self, image_url):
        try:
            payload = json.dumps({"image_url": image_url})  # ✅ Send URL instead of base64
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.primary_key}'
            }

            response = requests.post(self.scoring_uri, headers=headers, data=payload)

            if response.status_code == 200:
                response_data = response.json()  # ✅ No need for json.loads()

                if "coordinates" in response_data:
                    print("Coordinates:", response_data["coordinates"])
                else:
                    print("No coordinates found in response.")

                if "annotated_image" in response_data:
                    annotated_image_data = base64.b64decode(response_data["annotated_image"])

                    original_image_name = os.path.splitext(os.path.basename(image_url))[0]
                    annotated_image_path = f"{original_image_name}_annotated.jpg"

                    with open(annotated_image_path, "wb") as image_file:
                        image_file.write(annotated_image_data)

                    return annotated_image_path
                else:
                    print("Error in response:", response_data.get("error"))
            else:
                print(f"Request failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error annotating image: {e}")
        
        return None

    def process_image(self, image_path):
        original_image_url = self.upload_image_to_blob(image_path)
        if not original_image_url:
            print("Failed to upload image to Azure Blob Storage.")
            return

        print(f"Image uploaded to Azure Blob Storage: {original_image_url}")

        annotated_image_path = self.annotate_image(original_image_url)  # ✅ Pass URL instead of local file
        if not annotated_image_path:
            print("Failed to annotate image.")
            return

        annotated_image_url = self.upload_image_to_blob(annotated_image_path)
        if not annotated_image_url:
            print("Failed to upload annotated image to Azure Blob Storage.")
            return

        print(f"Annotated image uploaded to Azure Blob Storage: {annotated_image_url}")

if __name__ == "__main__":
    connection_string = "DefaultEndpointsProtocol=https;AccountName=dronetjekmlmod0588435650;AccountKey=RMGc4yrlczOZolE3MhWsXEN8bZlJHZtEZFEbUFk9xnm+/zD4EcDScSCUKTfVqm/wVSHJlKqfdqIJ+ASt8vHzLg==;EndpointSuffix=core.windows.net"
    container_name = "azureml"
    scoring_uri = "https://drontjek-yolo-model-f4ccffdte9gubtgx.northeurope-01.azurewebsites.net/predict"  # ✅ Updated
    primary_key = "C3eNtCJkShvkHncPHiGB1sheElXVIyYm"
    image_path = "roof009.jpg"  # Replace with your image file

    annotator = ImageAnnotator(connection_string, container_name, scoring_uri, primary_key)
    annotator.process_image(image_path)
