#!/usr/bin/env bash
# Seeed reBot Arm B601-DM description (URDF + meshes) into vendor/rebot_b601/DM.
# The upstream repository has no license file, so it is fetched rather than redistributed here.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone --quiet --filter=blob:none https://github.com/Seeed-Projects/reBotArm_control_py "$tmp/repo"
git -C "$tmp/repo" checkout --quiet d4f3591
mkdir -p vendor/rebot_b601
rm -rf vendor/rebot_b601/DM
cp -r "$tmp/repo/urdf/DM" vendor/rebot_b601/
echo "vendor/rebot_b601/DM ready"
