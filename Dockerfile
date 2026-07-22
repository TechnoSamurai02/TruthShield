FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV COMMUNITY_FORENSICS_REPO_PATH=/app/vendor/Community-Forensics
ENV COMMUNITY_FORENSICS_MODEL_ID=OwensLab/commfor-model-224
ENV AI_IMAGE_DETECTOR_MODELS=/app/training/models/truthshield-image-detector-v4-comparison
ENV AI_IMAGE_FUSION_MODEL_PATH=/app/training/models/truthshield-image-fusion-v4.joblib
ENV AI_MANIPULATION_LOCALIZER_PATH=/app/training/models/truthshield-manipulation-localizer-v4
ENV MEDIA_DECISION_POLICY_PATH=/app/training/models/media-policy-v4.image.calibrated.json
ENV VIDEO_ANALYSIS_MODE=adaptive
ENV VIDEO_KEYFRAME_MAX=64
ENV VIDEO_WINDOW_MAX=8
ENV IMAGE_TRANSFORMATION_CHECKS=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

RUN git clone https://github.com/JeongsooP/Community-Forensics.git /app/vendor/Community-Forensics \
    && git -C /app/vendor/Community-Forensics checkout ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4 \
    && test "$(git -C /app/vendor/Community-Forensics rev-parse HEAD)" = "ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4" \
    && rm -rf /app/vendor/Community-Forensics/.git

RUN python -c "import sys; sys.path.insert(0, '/app/vendor/Community-Forensics'); from models import ViTClassifier; ViTClassifier.from_pretrained('OwensLab/commfor-model-224', device='cpu')"

COPY backend /app/backend
COPY training/models /app/training/models

WORKDIR /app/backend

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
