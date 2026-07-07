# Stage 1: Build the React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy frontend configuration and install dependencies
COPY frontend/package.json ./
RUN npm install

# Copy frontend source and build the static assets
COPY frontend/ ./
RUN npm run build

# Stage 2: Create the Python Production Image
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python requirements and install them
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application source files
COPY app.py predict.py tf-cnn-model.keras ./

# Copy compiled React static assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
