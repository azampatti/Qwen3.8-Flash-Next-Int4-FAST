#!/usr/bin/env bash
# Install this model's b12x recipes + mod into Eugr's spark-vllm-docker, and pull the b12x image they were validated on.
#   ./eugr-setup.sh                         (expects ~/spark-vllm-docker)
#   EUGR_DIR=/other/path ./eugr-setup.sh
# Re-run it after every 'git pull' here to update the copies there.
set -euo pipefail
cd "$(dirname "$0")"

EUGR_DIR="${EUGR_DIR:-$HOME/spark-vllm-docker}"
MOD=flashnext-int4-b12x
RECIPES=(qwen3.8-flash-next-int4-b12x-solo.yaml qwen3.8-flash-next-int4-b12x.yaml)
# The b12x image, pinned by digest: Eugr's build of 2026-09-13 (vLLM 0.1.dev20759). His "latest" moves; the 2026-09-21
# build is not a drop-in for this model yet. Tagged as vllm-node-b12x, the name the recipes use.
IMAGE_TAG=vllm-node-b12x
IMAGE_PIN=eugr/spark-vllm-b12x@sha256:8e7e062186f841453ef0ec6f713043c5b65447decc3835206685128c18e42262

if [ ! -x "$EUGR_DIR/run-recipe.sh" ] || [ ! -d "$EUGR_DIR/mods" ] || [ ! -d "$EUGR_DIR/recipes" ]; then
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
if docker image inspect "$IMAGE_TAG" --format '{{json .RepoDigests}}' 2>/dev/null | grep -q "${IMAGE_PIN#*@}"; then
  echo "Image $IMAGE_TAG already is the pinned build (${IMAGE_PIN#*@sha256:})"
else
  echo "Pulling the pinned b12x image (~25 GB the first time) ..."
  docker pull "$IMAGE_PIN"
  docker tag "$IMAGE_PIN" "$IMAGE_TAG"
  echo "Tagged it as $IMAGE_TAG"
fi

rm -rf "$EUGR_DIR/mods/$MOD"
cp -r "$MOD" "$EUGR_DIR/mods/$MOD"
cp "${RECIPES[@]}" "$EUGR_DIR/recipes/"

echo "Image: $IMAGE_TAG = $IMAGE_PIN"
echo "  Do not run build-and-copy.sh --exp-b12x afterwards: it re-pulls Eugr's 'latest' over this tag."
echo "Installed into $EUGR_DIR:"
echo "  mods/$MOD/"
for r in "${RECIPES[@]}"; do echo "  recipes/$r"; done
echo
echo "Run it from $EUGR_DIR:"
echo "  cd $EUGR_DIR"
echo "  ./run-recipe.sh recipes/qwen3.8-flash-next-int4-b12x-solo.yaml --solo   # one DGX Spark"
echo "  ./run-recipe.sh recipes/qwen3.8-flash-next-int4-b12x.yaml               # two-node cluster (run on the head node;"
echo "                                                                          #  the model must be downloaded on both nodes)"
echo "For the cluster, run this script on BOTH nodes (or copy the image: ./build-and-copy.sh -t $IMAGE_TAG --no-build -c)."
