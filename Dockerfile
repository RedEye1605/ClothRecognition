# Use Python 3.9
FROM python:3.9

# Install system dependencies for OpenCV
USER root
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy requirements from backend
COPY --chown=user backend/requirements.txt $HOME/app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY --chown=user backend $HOME/app/backend

# Change working directory to backend so 'app' module is found
WORKDIR $HOME/app/backend

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
