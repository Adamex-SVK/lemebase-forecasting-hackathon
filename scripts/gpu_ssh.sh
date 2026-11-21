#!/usr/bin/env bash
# Run a command on the GPU machine over the Cloudflare tunnel with the dedicated key (never a password).
#   GPU_SSH_HOST=<tunnel host from the organizers> bash scripts/gpu_ssh.sh 'nvidia-smi'
# Needs: `brew install cloudflared`, the key ~/.ssh/hackathon_gpu whose public half is in the machine's authorized_keys
# (see iterations/03-foundation-model/gpu-machine.md). The host name is a secret-ish, changing value: do not commit it.
: "${GPU_SSH_HOST:?set GPU_SSH_HOST to the tunnel host name}"
KEY="${GPU_SSH_KEY:-$HOME/.ssh/hackathon_gpu}"; USER_="${GPU_SSH_USER:-hack}"
exec ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  -o PreferredAuthentications=publickey -o ConnectTimeout=30 -o ServerAliveInterval=30 \
  -o ProxyCommand="cloudflared access ssh --hostname %h" "$USER_@$GPU_SSH_HOST" "$@"
