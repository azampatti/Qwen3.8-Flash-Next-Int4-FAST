#!/usr/bin/env bash
# Pull Eugr's latest b12x image, tag it vllm-node-b12x-<image build date, YYYYMMDD>, and install this model's b12x recipes +
# mods into Eugr's spark-vllm-docker with their container: line set to that tag.
#   ./eugr-setup.sh                         (expects ~/spark-vllm-docker)
#   EUGR_DIR=/other/path ./eugr-setup.sh
#   ./eugr-setup.sh --pull-only             (only the image; setup.sh uses this to decide which image to serve with)
# setup.sh runs this for you when Eugr's repo is there. Re-run it after every 'git pull' here to update the copies there.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env   # EUGR_DIR, B12X_SOURCE, B12X_IMAGE_FILE

PULL_ONLY=0; [ "${1:-}" = "--pull-only" ] && PULL_ONLY=1
MOD=flashnext-int4-b12x
FST_MOD=fst-ep-local         # per-node fastsafetensors loading for the cluster recipe (see fst-ep-local/run.sh)
RECIPES=(qwen3.8-flash-next-int4-b12x-solo.yaml qwen3.8-flash-next-int4-b12x.yaml)
# every recipe of ours whose container: line follows the pulled tag (the Swift1.5 ones only if swift15/setup.sh installed them)
OUR_RECIPES=("${RECIPES[@]}" qwen3.8-flash-next-swift15-b12x-solo.yaml qwen3.8-flash-next-swift15-b12x.yaml)

if ! eugr_present; then
  echo "Eugr's spark-vllm-docker was not found at $EUGR_DIR" >&2
  echo "Get it first:" >&2
  echo "  git clone https://github.com/eugr/spark-vllm-docker ~/spark-vllm-docker" >&2
  echo "then run this script again." >&2
  exit 1
fi

for f in "$MOD" "$FST_MOD" "${RECIPES[@]}"; do
  [ -e "$f" ] || { echo "Missing $f in $(pwd) -- run 'git pull' in this folder first." >&2; exit 1; }
done

command -v docker >/dev/null || { echo "docker is not installed" >&2; exit 1; }
echo "Pulling Eugr's latest b12x image $B12X_SOURCE (~18 GB the first time) ..."
if docker pull "$B12X_SOURCE"; then
  BUILT="$(docker image inspect -f '{{.Created}}' "$B12X_SOURCE")"          # RFC 3339, UTC
  B12X_IMAGE="vllm-node-b12x-$(printf '%s' "${BUILT:0:10}" | tr -d -)"
  docker tag "$B12X_SOURCE" "$B12X_IMAGE"
  echo "Tagged it as $B12X_IMAGE (built ${BUILT:0:10}, $(docker image inspect -f '{{index .RepoDigests 0}}' "$B12X_SOURCE"))"
elif [ -n "$B12X_IMAGE" ] && docker image inspect "$B12X_IMAGE" >/dev/null 2>&1; then
  echo "WARNING: could not pull $B12X_SOURCE -- keeping the local $B12X_IMAGE" >&2
else
  echo "Could not pull $B12X_SOURCE and there is no local vllm-node-b12x-YYYYMMDD image" >&2; exit 1
fi
printf '%s\n' "$B12X_IMAGE" > "$B12X_IMAGE_FILE"
[ "$PULL_ONLY" = 1 ] && exit 0

for m in "$MOD" "$FST_MOD"; do
  rm -rf "$EUGR_DIR/mods/$m"
  cp -r "$m" "$EUGR_DIR/mods/$m"
done
cp "${RECIPES[@]}" "$EUGR_DIR/recipes/"
for r in "${OUR_RECIPES[@]}"; do
  if [ -f "$EUGR_DIR/recipes/$r" ]; then sed -i "s/^container: .*/container: $B12X_IMAGE/" "$EUGR_DIR/recipes/$r"; fi
done

echo "Image: $B12X_IMAGE (every recipe of ours in $EUGR_DIR/recipes uses it)"
echo "Installed into $EUGR_DIR:"
echo "  mods/$MOD/"
echo "  mods/$FST_MOD/"
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
