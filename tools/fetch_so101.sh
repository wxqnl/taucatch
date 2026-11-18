#!/usr/bin/env bash
# SO-101 description (MJCF + meshes, Apache-2.0) from TheRobotStudio/SO-ARM100 into vendor/so101.
# The meshes are 19 MB, so they are fetched rather than kept in this repository.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone --quiet --filter=blob:none --no-checkout https://github.com/TheRobotStudio/SO-ARM100 "$tmp/repo"
git -C "$tmp/repo" sparse-checkout set Simulation/SO101 LICENSE
git -C "$tmp/repo" checkout --quiet a758567
rm -rf vendor/so101
mkdir -p vendor/so101
cp -r "$tmp/repo/Simulation/SO101/." vendor/so101/
cp "$tmp/repo/LICENSE" vendor/so101/LICENSE
echo a758567 > vendor/so101/UPSTREAM_COMMIT
echo "vendor/so101 ready"
