FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG GITHUB_TOKEN

ENV TZ=Europe/Madrid
ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
ENV MALLET_HOME=/opt/mallet
ENV PATH="$MALLET_HOME/bin:$PATH"

# Install system dependencies and ensure Java 21 compatibility
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
    tzdata zlib1g-dev libgl1-mesa-glx libgl1-mesa-dev libpython3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache/*

# Set Java 21 as the default
RUN update-alternatives --set java /usr/lib/jvm/java-21-openjdk-amd64/bin/java

# Verify Java version (should be Java 21)
RUN java -version

# Upgrade pip inside the virtual environment
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && pip cache purge

# Set working directory
WORKDIR /app

# Copy only requirements file first to optimize caching
COPY requirements.txt .

# Install project dependencies inside the virtual environment
RUN pip install --upgrade pip setuptools wheel

RUN git clone https://x-access-token:${GITHUB_TOKEN}@github.com/lcalvobartolome/portalia-pipeline.git /opt/alia-pipeline && \
    pip install --no-cache-dir --ignore-installed -e /opt/alia-pipeline && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt && \
    pip install --ignore-installed \
        https://github.com/explosion/spacy-models/releases/download/es_core_news_lg-3.8.0/es_core_news_lg-3.8.0-py3-none-any.whl && \
    pip cache purge
    
# Mallet
# Download and install Mallet (removing unnecessary files to save space)
#RUN wget $MALLET_URL -O /tmp/mallet.tar.gz && \
#    tar -xvzf /tmp/mallet.tar.gz -C /opt && \
#    mv /opt/Mallet-${MALLET_VERSION} /opt/mallet && \
#    rm /tmp/mallet.tar.gz && \
#    rm -rf /opt/mallet/sample-data  # Remove sample data to reduce image size
COPY mallet-2.0.8/ /opt/mallet/

RUN mkdir -p /mnt/data /data/inference

# Pre-descargar el modelo de embeddings para no descargarlo en runtime
ARG MODEL_NAME=hiiamsid/sentence_similarity_spanish_es
RUN python3 - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer('${MODEL_NAME}').save('/tmp/huggingface')
EOF

COPY . .

EXPOSE 10083
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10083", "--workers", "2"]