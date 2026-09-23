#!/usr/bin/env bash
# Install this model's b12x recipes + mod into Eugr's spark-vllm-docker.
#   ./eugr-setup.sh                         (expects ~/spark-vllm-docker)
#   EUGR_DIR=/other/path ./eugr-setup.sh
# Re-run it after every 'git pull' here to update the copies there.
set -euo pipefail
cd "$(dirname "$0")"

EUGR_DIR="${EUGR_DIR:-$HOME/spark-vllm-docker}"
MOD=flashnext-int4-b12x
RECIPES=(qwen3.8-flash-next-int4-b12x-solo.yaml qwen3.8-flash-next-int4-b12x.yaml)

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

rm -rf "$EUGR_DIR/mods/$MOD"
cp -r "$MOD" "$EUGR_DIR/mods/$MOD"
cp "${RECIPES[@]}" "$EUGR_DIR/recipes/"

echo "Installed into $EUGR_DIR:"
echo "  mods/$MOD/"
for r in "${RECIPES[@]}"; do echo "  recipes/$r"; done
echo
echo "Run it from $EUGR_DIR:"
echo "  cd $EUGR_DIR"
echo "  ./run-recipe.sh recipes/qwen3.8-flash-next-int4-b12x-solo.yaml --solo   # one DGX Spark"
echo "  ./run-recipe.sh recipes/qwen3.8-flash-next-int4-b12x.yaml               # two-node cluster (run on the head node;"
echo "                                                                          #  the model must be downloaded on both nodes)"
