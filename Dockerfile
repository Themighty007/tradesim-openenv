FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
# Start the server with C-level uvloop and 2 workers
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--loop", "uvloop"]