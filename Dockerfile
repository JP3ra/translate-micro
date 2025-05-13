# Use an official Python image (choose GPU-friendly image if you need GPU)
FROM python:3.10-slim

# Optional: Install OS-level dependencies if needed
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy your entire code into the container
COPY . /app

# Make your script executable
RUN chmod +x translate-micro-build.sh

# Run your bash script to install everything
RUN ./translate-micro-build.sh

# Expose the port Flask runs on
EXPOSE 5000

# Start your app
CMD ["python", "app.py"]
