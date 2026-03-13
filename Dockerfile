FROM python:3.12

WORKDIR /app

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python dependencies first (better build cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default port inside the container; Koyeb will override PORT via environment variable
ENV PORT=8000

# Use Gunicorn to serve the Flask app; bind to $PORT provided by Koyeb
CMD ["sh", "-c", "gunicorn app.app:app --bind 0.0.0.0:${PORT:-8000} --workers 2 --threads 4 --timeout 120 --graceful-timeout 30 --keep-alive 5"]

