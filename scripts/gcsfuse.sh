#!/usr/bin/env bash
set -eux pipefail
export fuse_user=$USER
sudo --preserve-env=fuse_user GOOGLE_APPLICATION_CREDENTIALS=~/Nextcloud/Documents/fafa-vae-0102b4da1611.json \
  gcsfuse -o allow_other \
  --uid $(id -u "$fuse_user") \
  --gid $(id -g "$fuse_user") \
  --debug_gcs \
  --implicit-dirs \
  antfield ~/mnt/antfield
touch ~/mnt/antfield/test.txt
