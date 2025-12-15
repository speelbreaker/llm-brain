FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN mkdir -p /var/lib/pr_supervisor/jobs /var/lib/pr_supervisor/workspaces

ENV PYTHONPATH=/app
ENV SUPERVISOR_BASE_JOBS_DIR=/var/lib/pr_supervisor/jobs
ENV SUPERVISOR_STORE_PATH=/var/lib/pr_supervisor/store.jsonl

EXPOSE 8080

CMD ["uvicorn", "src.supervisor.app:app", "--host", "0.0.0.0", "--port", "8080"]
