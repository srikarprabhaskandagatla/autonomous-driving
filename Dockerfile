FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

COPY src/ ./src/
COPY weights/ ./weights/

EXPOSE 8080
EXPOSE 9090

CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8080"]
