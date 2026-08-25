# sia-config

Configuration bind-mounted into the `sia-core-api` container at `/config` (see
`sia-core-api` volumes in `docker-compose.yaml` / `docker-compose.podman.yaml`
and `API_KEYS_FILE`/config path defaults below). Changes here take effect on
container restart; no rebuild needed.