FROM python:3.12-slim-bookworm
WORKDIR /app

# Install only runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libffi8 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy API code
COPY . .

RUN mkdir -p /data/source
RUN mkdir -p /data/inference

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "dask[dataframe]"

EXPOSE 10083
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10083", "--workers", "2"]