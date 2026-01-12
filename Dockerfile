# Use Python 3.9
FROM python:3.9

# Set working directory to /code
WORKDIR /code

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy requirements from backend
COPY backend/requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy backend (which now includes models in backend/models)
COPY backend /code/backend

# Change working directory to backend so 'app' module is found
WORKDIR /code/backend

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
