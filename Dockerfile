FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AI_IMAGE_DETECTOR_MODELS=/app/training/models/truthshield-image-detector-v2
ENV AI_MANIPULATION_LOCALIZER_PATH=/app/training/models/truthshield-manipulation-localizer-v4
ENV MEDIA_DECISION_POLICY_PATH=/app/training/models/media-policy-v4.json
ENV VIDEO_ANALYSIS_MODE=adaptive
ENV VIDEO_KEYFRAME_MAX=64
ENV VIDEO_WINDOW_MAX=8
ENV IMAGE_TRANSFORMATION_CHECKS=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY training/models /app/training/models

WORKDIR /app/backend

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
