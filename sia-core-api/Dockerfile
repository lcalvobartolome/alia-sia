FROM nvidia/cuda:12.1.1-base-ubuntu22.04
ARG GITHUB_TOKEN
ARG TARGETARCH
ENV TZ=Europe/Madrid
ENV DEBIAN_FRONTEND=noninteractive
ENV MALLET_VERSION=202108
ENV MALLET_URL="https://github.com/mimno/Mallet/releases/download/v${MALLET_VERSION}/Mallet-${MALLET_VERSION}-bin.tar.gz"
ENV MALLET_HOME=/opt/mallet
ENV PATH="$MALLET_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk \
    poppler-utils \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git ant wget curl \
    libbz2-dev libffi-dev libgdbm-dev libncurses5-dev \
    libnss3-dev libreadline-dev libsqlite3-dev libssl-dev \
    tzdata zlib1g-dev libpython3-dev \
    gcc g++ build-essential cmake \
    ca-certificates openssl && \
    update-ca-certificates && \
    rm -f /usr/lib/aarch64-linux-gnu/libGLX_indirect.so.0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache/*

# Set JAVA_HOME dynamically based on architecture
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        JAVA_PATH=/usr/lib/jvm/java-21-openjdk-arm64; \
    else \
        JAVA_PATH=/usr/lib/jvm/java-21-openjdk-amd64; \
    fi && \
    update-alternatives --set java ${JAVA_PATH}/bin/java && \
    echo "JAVA_HOME=${JAVA_PATH}" >> /etc/environment && \
    echo "export JAVA_HOME=${JAVA_PATH}" >> /etc/profile.d/java.sh && \
    ln -sf ${JAVA_PATH} /usr/lib/jvm/java-21-openjdk-current

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-current

# Verify Java version
RUN java -version

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && pip cache purge

# Set working directory
WORKDIR /app

# Copy only requirements file first to optimize caching
COPY requirements.txt .

# Ensure pip uses updated certificates
RUN pip config set global.cert /etc/ssl/certs/ca-certificates.crt

# Install project dependencies
RUN pip install --upgrade pip setuptools wheel

RUN git clone https://x-access-token:${GITHUB_TOKEN}@github.com/lcalvobartolome/portalia-pipeline.git /opt/alia-pipeline && \
    pip install --no-cache-dir --ignore-installed -e /opt/alia-pipeline && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt && \
    pip install --ignore-installed \
        https://github.com/explosion/spacy-models/releases/download/es_core_news_lg-3.8.0/es_core_news_lg-3.8.0-py3-none-any.whl && \
    pip cache purge

# Mallet
RUN wget $MALLET_URL -O /tmp/mallet.tar.gz && \
    tar -xvzf /tmp/mallet.tar.gz -C /opt && \
    mv /opt/Mallet-${MALLET_VERSION} /opt/mallet && \
    rm /tmp/mallet.tar.gz && \
    rm -rf /opt/mallet/sample-data  # Remove sample data to reduce image size

RUN mkdir -p /mnt/data /data/inference

# Persistent Hugging Face cache baked into the image itself
ENV HF_HOME=/opt/hf_cache
ENV TRANSFORMERS_CACHE=/opt/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/opt/hf_cache

# Pre-download the embeddings model into that cache
ARG MODEL_NAME=hiiamsid/sentence_similarity_spanish_es
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

# Fail fast instead of silently trying (and hanging) to hit the network if the cache is ever missing something once the container is moved/offline.
ENV HF_HUB_OFFLINE=1

COPY . .
EXPOSE 10083
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10083", "--workers", "2"]