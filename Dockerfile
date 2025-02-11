FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20240418.v1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev

# Set the working directory
WORKDIR /usr/src/app

# Copy requirements.txt to the working directory
COPY requirements.txt ./

# Install Python dependencies directly
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code to the working directory
COPY YOLOAnnotator.py .
COPY yolo-v11.pt .  


# Expose port 8000 for the application
EXPOSE 8000

# Set the default command to run the application with uvicorn
CMD ["uvicorn", "YOLOAnnotator:app", "--host", "0.0.0.0", "--port", "8000"]
