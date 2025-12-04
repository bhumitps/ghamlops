FROM python:3.11-slim

# Helpful env defaults
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Working dir inside the container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the rest of the app (app.py, etc.)
COPY . .

# Hugging Face Spaces sets PORT (usually 7860); we respect it.
ENV PORT=7860
EXPOSE 7860

# Start Streamlit directly on 0.0.0.0:7860 in headless mode
CMD ["bash", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]
