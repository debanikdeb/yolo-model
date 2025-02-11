#!/bin/bash

# Install system dependencies (including GLib and Mesa for OpenGL)
apt-get update && \
apt-get install -y libgl1-mesa-dev libglib2.0-0 libglib2.0-dev libsm6 libxrender1 libxext6

# Install Python dependencies
pip install -r requirements.txt

# Start the application using Uvicorn
uvicorn YOLOAnnotator:app --host 0.0.0.0 --port 8000
