#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# helpers
wait_for_port() {
  local host=$1 port=$2 service=$3
  echo "Esperando a que $service esté listo en $host:$port..."
  until nc -z "$host" "$port" 2>/dev/null; do
    echo "  $service no disponible, reintentando en 2s..."
    sleep 2
  done
  echo "  $service listo."
}

wait_for_http() {
  local url=$1 service=$2
  echo "Esperando a que $service responda en $url..."
  until curl -sf "$url" > /dev/null 2>&1; do
    echo "  $service no responde aún, reintentando en 3s..."
    sleep 3
  done
  echo "  $service listo."
}

remove_if_exists() {
  local name=$1
  if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
    echo "Eliminando contenedor existente: $name"
    docker rm -f "$name"
  fi
}

# crate net
docker network create sia-net 2>/dev/null || echo "Red sia-net ya existe."

# create directories
mkdir -p ./db/data/solr
mkdir -p ./db/data/zoo/data
mkdir -p ./db/data/zoo/logs
mkdir -p ./db/data/sqlite3
mkdir -p ./data

if [ ! -f ./db/data/sqlite3/pipeline_jobs.db ]; then
  touch ./db/data/sqlite3/pipeline_jobs.db
fi

# chown inside a container so the UID maps correctly (uses solr image already required, runs as root)
docker run --rm \
  --user root \
  --entrypoint chown \
  -v "$(pwd)/db/data/solr:/var/solr" \
  solr:9.1.1 \
  -R 8983:8983 /var/solr

# remove existing containers if they exist
remove_if_exists sia-zoo
remove_if_exists sia-solr
remove_if_exists sia-core-api

# zoo
echo ""
echo "Arrancando Zookeeper..."
docker run -d \
  --name sia-zoo \
  --network sia-net \
  --network-alias zoo \
  --restart always \
  -p 10086:8080 \
  -p 10087:2181 \
  -e JVMFLAGS=-Djute.maxbuffer=50000000 \
  -v "$(pwd)/db/data/zoo/data:/data" \
  -v "$(pwd)/db/data/zoo/logs:/datalog" \
  --memory 100g \
  zookeeper:latest

wait_for_port localhost 10087 "Zookeeper"

# solr
JAR="./solr-plugins/NP-solr-dist-plugin/NP-solr-dist-plugin.jar"
if [ ! -f "$JAR" ]; then
  echo "ERROR: El plugin de Solr no existe o no es un fichero: $JAR"
  echo "  Descárgalo o transfiere el JAR antes de continuar."
  exit 1
fi

echo ""
echo "Arrancando Solr..."
docker run -d \
  --name sia-solr \
  --network sia-net \
  --restart always \
  -p 10085:8983 \
  -v "$(pwd)/db/data/solr:/var/solr" \
  -v "$(pwd)/solr-plugins/NP-solr-dist-plugin/NP-solr-dist-plugin.jar:/opt/solr/dist/plugins/NP-solr-dist-plugin.jar" \
  -v "$(pwd)/solr-config:/opt/solr/server/solr" \
  -v "$(pwd)/solr-config/solr.in.sh:/etc/default/solr.in.sh" \
  -e 'SOLR_OPTS=-Dsolr.jetty.request.header.size=65535' \
  -e 'SOLR_JAVA_MEM=-Xms1g -Xmx1g' \
  --memory 100g \
  --entrypoint docker-entrypoint.sh \
  solr:9.1.1 \
  solr start -f -c -z zoo:2181 \
  -a "-Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=n,address=1044 -Djute.maxbuffer=0x5000000"

wait_for_http "http://localhost:10085/solr/admin/info/system" "Solr"

# init solr config
echo ""
echo "Inicializando configuración de Solr..."
chmod +x solr-config/bash_scripts/init_config.sh
solr-config/bash_scripts/init_config.sh ./db/data/solr/data

# sia-core-api
GPU_FLAGS=()
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
  echo "GPU NVIDIA detectada, activando soporte GPU (device=2)..."
  GPU_FLAGS=(--gpus "device=2" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
else
  echo "Sin GPU NVIDIA, arrancando sin soporte GPU."
fi

echo ""
echo "Arrancando sia-core-api..."
docker run -d \
  --name sia-core-api \
  --network sia-net \
  -p 10083:10083 \
  -e SOLR_URL=http://solr:8983 \
  -e "SIA_MASTER_KEY=${SIA_MASTER_KEY:-master-key-change-in-production}" \
  -e API_KEYS_FILE=/config/api_keys.json \
  -e "CORS_ORIGINS=${CORS_ORIGINS:-http://localhost:3000,http://localhost:8080}" \
  --add-host host.docker.internal:host-gateway \
  -v "$(pwd)/data/:/mnt/data" \
  -v "$(pwd)/sia-config:/config" \
  -v "$(pwd)/db/data/sqlite3/pipeline_jobs.db:/data/pipeline_jobs.db" \
  --memory 100g \
  "${GPU_FLAGS[@]}" \
  portalia-sia-sia-core-api:latest

echo ""
echo "Todos los servicios arrancados."
echo "  Zookeeper:   http://localhost:10086"
echo "  Solr:        http://localhost:10085/solr"
echo "  sia-core-api: http://localhost:10083"
