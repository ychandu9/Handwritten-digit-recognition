# Deployment Guide: Split Architecture (Vercel + Render)

This guide details how to deploy the application with a split architecture:
- **React Frontend**: Deployed on **Vercel** (for high speed, global edge network, and free static hosting).
- **FastAPI Backend**: Deployed on **Render** (as a Web Service to host the Python & TensorFlow/Keras API).

---

## 🎨 Step 1: Deploy Backend to Render

1. **Push Code to Git**:
   Ensure all changes are committed and pushed to your Git repository.
2. **Create Web Service on Render**:
   - Go to [Render Dashboard](https://dashboard.render.com) and click **New** -> **Web Service**.
   - Select your Git repository.
3. **Configure Service Details**:
   - **Name**: `digit-recognition-backend`
   - **Environment / Runtime**: **Python 3** (or Docker if you prefer, but Python is simpler)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: **Free** (or Starter)
4. **Deploy & Copy URL**:
   - Click **Deploy Web Service**.
   - Once the deploy completes, copy your service's URL (e.g., `https://digit-recognition-backend.onrender.com`).

---

## 🚀 Step 2: Deploy Frontend to Vercel

1. **Create Project on Vercel**:
   - Go to [Vercel Dashboard](https://vercel.com/dashboard).
   - Click **Add New...** -> **Project**.
   - Import your Git repository.
2. **Configure Directory and Framework**:
   - **Framework Preset**: Vercel will automatically detect **Vite**.
   - **Root Directory**: Click "Edit" and choose the `frontend` folder.
3. **Set Environment Variables**:
   - Expand the **Environment Variables** section.
   - Add a new variable:
     - **Key**: `VITE_API_URL`
     - **Value**: `https://your-backend-url.onrender.com/api/predict` (replace with the Render backend URL you copied in Step 1, making sure to append `/api/predict` at the end).
4. **Deploy**:
   - Click **Deploy**.
   - Vercel will build the React application and provide you with a public URL for your UI.

---

## 🐳 Alternative: Unified Deployment (Render only)

If you prefer to run both backend and frontend inside a single service (on Render only):
- Choose **Docker** as the runtime on Render.
- Render will automatically use the `Dockerfile` in the root to build the React code and run the FastAPI server, hosting everything under a single domain on port `8000`.
