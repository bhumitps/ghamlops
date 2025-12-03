# Use a lightweight Python image
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install -r tourism_project/deployment/requirements.txt

# Expose the port used by Streamlit
EXPOSE 7860

# Streamlit specific environment variables (optional tweaks)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default command to run the Streamlit app
CMD ["streamlit", "run", "tourism_project/deployment/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
