FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir pydantic openai numpy pytest
ENV PYTHONPATH=/app
CMD ["python", "src/agent/inference.py"]
