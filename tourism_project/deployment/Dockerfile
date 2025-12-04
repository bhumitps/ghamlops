FROM python:3.11-slim

# Helpful env defaults
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Working dir inside the container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the app (app.py, etc.)
COPY . /app

# Hugging Face Docker Spaces expose port 7860; we listen on it directly
EXPOSE 7860

# Start Streamlit directly (no bash, use exec form)
CMD ["streamlit", "run", "app.py",
     "--server.port=7860",
     "--server.address=0.0.0.0",
     "--server.headless=true",
     "--server.enableCORS=false",
     "--server.enableXsrfProtection=false"]
