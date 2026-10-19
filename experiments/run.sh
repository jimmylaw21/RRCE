#!/usr/bin/env bash
set -euo pipefail
task="${1:?Specify nudity, dog, or van_gogh}"
stage="${2:?Specify manifests, retain, gallery, train, or evaluate}"
shift 2
case "$task" in
    nudity|dog|van_gogh) ;;
    *) printf 'Unknown task: %s\n' "$task" >&2; exit 2 ;;
esac
case "$stage" in
    manifests|retain|gallery) python prepare_data.py --config "configs/$task.json" --stage "$stage" "$@" ;;
    train) python train.py --config "configs/$task.json" "$@" ;;
    evaluate) python evaluate.py --config "configs/$task.json" "$@" ;;
    *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac
