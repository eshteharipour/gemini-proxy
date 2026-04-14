# ── Build ────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Choose to use v1 or v2
# COPY main_v1.py main.py
COPY main_v2.py main.py

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000
CMD ["python", "main.py"]
