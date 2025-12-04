FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 7860

CMD ["sh", "-c", "echo '--- ENV DUMP START ---'; env | sort; echo '--- ENV DUMP END ---'; streamlit run app.py --server.port=7860 --server.address=0.0.0.0"]
