# Start from a PyTorch image with CUDA
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app
RUN mkdir /dataset/
RUN mkdir /models/
RUN apt-get -y update; apt-get -y install curl
# Copy the dependencies file to the working directory
COPY . .

# Install any dependencies
RUN pip install -r requirements.txt
