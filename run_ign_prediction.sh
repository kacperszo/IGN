#!/usr/bin/env bash
# Run IGN binding-affinity prediction on a prepared input directory.
#
# The directory must contain exactly one .zip laid out for mode2 (one subdirectory
# per target, each with its own .pdb and .sdf) — see
# MD_pretraining/evaluate/frames_to_ign.py, which builds it.
#
# IGN writes prediction.csv back into that same directory, and deletes everything
# there that is not a .csv or .zip, so give it a directory of its own.
#
#   ./run_ign_prediction.sh ~/md_data/ign_input
#   ./run_ign_prediction.sh ~/md_data/ign_input --num_process 6
#
# NOTE: the image pins torch 1.3.1, which predates `weights_only`. Loading the five
# checkpoints in model_save/ is therefore an unrestricted unpickle of third-party
# files. Scan them first:
#   uv run -m tools.scan_pickle model_save/    (from the MD_pretraining repo)

set -euo pipefail

IMAGE="${IGN_IMAGE:-localhost/ign:latest}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$(command -v podman || command -v docker)"

if [ $# -lt 1 ]; then
    echo "usage: $0 <input_dir> [extra args passed to model_ign_prediction.py]" >&2
    exit 1
fi

INPUT_DIR="$(cd "$1" && pwd)"
shift

zip_count=$(find "$INPUT_DIR" -maxdepth 1 -name '*.zip' | wc -l)
if [ "$zip_count" -ne 1 ]; then
    echo "error: expected exactly one .zip in $INPUT_DIR, found $zip_count" >&2
    exit 1
fi

if ! "$ENGINE" image exists "$IMAGE" 2>/dev/null; then
    echo "=== building $IMAGE (first run only) ==="
    "$ENGINE" build -t "$IMAGE" -f "$REPO_DIR/Containerfile" "$REPO_DIR"
fi

echo "=== running IGN on $INPUT_DIR ==="
echo "    image:  $IMAGE  (CPU only — the image installs torch 1.3.1+cpu)"

# --userns=keep-id maps the container user onto the host user, without it a rootless
# engine writes prediction.csv as a subuid the host cannot read
"$ENGINE" run --rm \
    --userns=keep-id \
    -v "$INPUT_DIR":/data:z \
    -w /work/scripts \
    "$IMAGE" \
    python model_ign_prediction.py --test_file_path=/data "$@"

echo
if [ -f "$INPUT_DIR/prediction.csv" ]; then
    echo "=== done: $INPUT_DIR/prediction.csv ($(($(wc -l < "$INPUT_DIR/prediction.csv") - 1)) rows) ==="
    head -3 "$INPUT_DIR/prediction.csv"
else
    echo "!!! no prediction.csv produced — check the output above" >&2
    exit 1
fi
