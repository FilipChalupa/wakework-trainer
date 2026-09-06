#!/bin/sh
# Runs the app as an unprivileged user whose uid/gid can be set with PUID/PGID (default 1000),
# so files created in the /data bind mount belong to the host user instead of root.
set -e
PUID="${PUID:-1000}"
PGID="${PGID:-1000}"
if [ "$(id -u)" = "0" ]; then
    if [ "$(id -g app)" != "$PGID" ]; then groupmod -o -g "$PGID" app; fi
    if [ "$(id -u app)" != "$PUID" ]; then usermod -o -u "$PUID" app; fi
    mkdir -p /data
    chown app:app /data
    # Fix ownership of data created by older (root) versions of the image. Cheap when already correct.
    if find /data ! -user app -print -quit 2>/dev/null | grep -q .; then chown -R app:app /data; fi
    exec gosu app "$@"
fi
exec "$@"
