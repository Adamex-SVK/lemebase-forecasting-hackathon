#!/usr/bin/env bash
# Copy files to / from the GPU machine (key only).
#   GPU_SSH_HOST=<host> bash scripts/gpu_scp.sh up   dist/gpu_bundle.tar.gz '~/hack/gpu_bundle.tar.gz'
#   GPU_SSH_HOST=<host> bash scripts/gpu_scp.sh down '~/hack/forecasts.tgz' ./forecasts.tgz
: "${GPU_SSH_HOST:?set GPU_SSH_HOST to the tunnel host name}"
KEY="${GPU_SSH_KEY:-$HOME/.ssh/hackathon_gpu}"; USER_="${GPU_SSH_USER:-hack}"
O=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
   -o PreferredAuthentications=publickey -o ProxyCommand="cloudflared access ssh --hostname %h")
if [ "$1" = up ]; then exec scp "${O[@]}" "$2" "$USER_@$GPU_SSH_HOST:$3"; else exec scp "${O[@]}" "$USER_@$GPU_SSH_HOST:$2" "$3"; fi
