# Deployment Guide: FashionAI Migration

This guide explains how to deploy the backend to **Hugging Face Spaces** and the frontend to **Vercel**.

## 1. Backend Deployment (Hugging Face Spaces)

The backend handles the AI model processing. We will deploy it as a Docker Space.

1.  **Create a New Space**:
    - Go to [Hugging Face Spaces](https://huggingface.co/spaces).
    - Click **"Create new Space"**.
    - Enter a name (e.g., `clothing-recognition-backend`).
    - Select **Docker** as the SDK.
    - Choose **Public**.
    - Click **"Create Space"**.

2.  **Upload Files**:
    - You can upload files directly via the browser or use git.
    - **Crucial**: You need to upload the **Entire Project** (including `backend/`, `models/`, and the `Dockerfile` at the root).
    - If using the web interface, drag and drop the folders.
    - If using git (recommended):
      ```bash
      git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
      cp -r /path/to/local/project/* .
      git add .
      git commit -m "Initial commit"
      git push
      ```

3.  **Wait for Build**:
    - The Space will start building the Docker image.
    - Once "Running", copy the Direct URL (top right menu > Embed this space > Direct URL).
    - It usually looks like: `https://username-space-name.hf.space`.

## 2. Frontend Deployment (Vercel)

The frontend is the user interface.

1.  **Configure API URL**:
    - Open `frontend/app.html`.
    - Find the line `const API_URL = ...`.
    - Change it to your Hugging Face Space URL (e.g., `https://username-space-name.hf.space`).
    - **Note**: Ensure you use `https` and remove any trailing slash.

2.  **Deploy**:
    - Go to [Vercel Dashboard](https://vercel.com/dashboard).
    - Click **"Add New..."** > **"Project"**.
    - Import your GitHub repository (or upload the `frontend` folder).
    - **Root Directory**: Select `frontend` as the root directory if you imported the whole repo.
    - Click **Deploy**.

## Troubleshooting
- **CORS Errors**: The backend is already configured to allow all origins (`allow_origins=["*"]`), so it should work from Vercel.
- **Model Not Found**: Ensure the `models/` folder was uploaded to Hugging Face. The `Dockerfile` assumes `backend/` and `models/` are side-by-side in the root.

## Performance & Limitations (Hugging Face Free Tier)
- **Cold Starts**: Free Spaces "sleep" after 48 hours of inactivity. The first request may take 1-2 minutes to wake up the server. The frontend is optimized to handle this by retrying automatically.
- **CPU vs GPU**: This project runs on CPU by default. For faster inference, you can upgrade the Space to a GPU instance, but the current optimization is sufficient for standard usage.
- **Image Size**: The frontend automatically resizes images to 1024px before uploading to reduce latency and bandwidth.

