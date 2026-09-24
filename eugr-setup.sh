#!/usr/bin/env bash
# Install this model's b12x recipes + mod into Eugr's spark-vllm-docker, and pull the b12x image they were validated on.
#   ./eugr-setup.sh                         (expects ~/spark-vllm-docker)
#   EUGR_DIR=/other/path ./eugr-setup.sh
#   ./eugr-setup.sh --pull-only             (only the pinned image; setup.sh uses this to decide which image to serve with)
# setup.sh runs this for you when Eugr's repo is there. Re-run it after every 'git pull' here to update the copies there.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env   # EUGR_DIR, B12X_IMAGE, B12X_IMAGE_PIN

PULL_ONLY=0; [ "${1:-}" = "--pull-only" ] && PULL_ONLY=1
MOD=flashnext-int4-b12x
RECIPES=(qwen3.8-flash-next-int4-b12x-solo.yaml qwen3.8-flash-next-int4-b12x.yaml)

if ! eugr_present; then
  echo "Eugr's spark-vllm-docker was not found at $EUGR_DIR" >&2
  echo "Get it first:" >&2
  echo "  git clone https://github.com/eugr/spark-vllm-docker ~/spark-vllm-docker" >&2
  echo "then run this script again." >&2
  exit 1
fi

for f in "$MOD" "${RECIPES[@]}"; do
  [ -e "$f" ] || { echo "Missing $f in $(pwd) -- run 'git pull' in this folder first." >&2; exit 1; }
done

command -v docker >/dev/null || { echo "docker is not installed" >&2; exit 1; }
if docker image inspect "$B12X_IMAGE" --format '{{json .RepoDigests}}' 2>/dev/null | grep -q "${B12X_IMAGE_PIN#*@}"; then
  echo "Image $B12X_IMAGE already is the pinned build (${B12X_IMAGE_PIN#*@sha256:})"
else
  echo "Pulling the pinned b12x image (~25 GB the first time) ..."
  docker pull "$B12X_IMAGE_PIN"
  docker tag "$B12X_IMAGE_PIN" "$B12X_IMAGE"
  echo "Tagged it as $B12X_IMAGE"
fi
[ "$PULL_ONLY" = 1 ] && exit 0

rm -rf "$EUGR_DIR/mods/$MOD"
cp -r "$MOD" "$EUGR_DIR/mods/$MOD"
cp "${RECIPES[@]}" "$EUGR_DIR/recipes/"

echo "Image: $B12X_IMAGE = $B12X_IMAGE_PIN"
echo "  Do not run build-and-copy.sh --exp-b12x afterwards: it re-pulls Eugr's 'latest' over this tag."
echo "Installed into $EUGR_DIR:"
echo "  mods/$MOD/"
for r in "${RECIPES[@]}"; do echo "  recipes/$r"; done
echo
echo "Run it from this folder (settings in serve.sh):"
echo "  ./serve.sh                                                              # one DGX Spark, Eugr's launcher underneath"
echo "Or straight from $EUGR_DIR:"
echo "  cd $EUGR_DIR"
echo "  ./run-recipe.sh recipes/qwen3.8-flash-next-int4-b12x-solo.yaml --solo   # one DGX Spark"
echo "  ./run-recipe.sh recipes/qwen3.8-flash-next-int4-b12x.yaml               # two-node cluster (run on the head node;"
echo "                                                                          #  the model must be downloaded on both nodes)"
echo "For the cluster, run this script on BOTH nodes (or copy the image: ./build-and-copy.sh -t $B12X_IMAGE --no-build -c)."
