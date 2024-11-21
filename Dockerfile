FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /application

# Copy all files from the current directory to the working directory in the container
COPY . /application

# Update the package list and install necessary dependencies
RUN apt update -y && apt install -y awscli \
    && apt-get update && apt-get install -y ffmpeg libsm6 libxext6 unzip \
    && pip install --no-cache-dir -r requirements.txt

# Specify the default command to run the application
CMD ["python3", "application.py"]
