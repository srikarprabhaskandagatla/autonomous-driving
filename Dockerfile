FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Install Python dependencies first — Docker layer caching means
# this layer only rebuilds when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        git+https://github.com/facebookresearch/segment-anything.git

# Copy source code
COPY src/      ./src/
COPY weights/  ./weights/

EXPOSE 8080   
EXPOSE 9090   

# uvicorn with 1 worker — see serving/server.py for explanation
CMD ["python", "-m", "uvicorn", "src.serving.server:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]