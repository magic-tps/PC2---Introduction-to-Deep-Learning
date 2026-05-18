#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/data/dataset"

has_imagefolder_layout() {
  local candidate="$1"
  [ -n "$candidate" ] \
    && [ -d "$candidate/images/TRAIN" ] \
    && [ -d "$candidate/images/TEST" ] \
    && compgen -G "$candidate/images/TRAIN/*/*.jpeg" >/dev/null \
    && compgen -G "$candidate/images/TEST/*/*.jpeg" >/dev/null
}

if has_imagefolder_layout "$TARGET_DIR"; then
  echo "Dataset already available at $TARGET_DIR"
  exit 0
fi

LOCAL_CANDIDATES=(
  "${DATASET_DIR:-}"
  "$ROOT_DIR/../dataset2-master/dataset2-master"
  "$ROOT_DIR/../dataset2-master"
  "$ROOT_DIR/../dataset-master/dataset-master"
  "$ROOT_DIR/../dataset-master"
)

for candidate in "${LOCAL_CANDIDATES[@]}"; do
  if has_imagefolder_layout "$candidate"; then
    mkdir -p "$ROOT_DIR/data"
    TMP_DIR="$TARGET_DIR.tmp"
    rm -rf "$TMP_DIR"
    mkdir -p "$TMP_DIR"
    cp -R "$candidate/." "$TMP_DIR/"
    rm -rf "$TARGET_DIR"
    mv "$TMP_DIR" "$TARGET_DIR"
    echo "Copied dataset from $candidate to $TARGET_DIR"
    exit 0
  fi
done

cat <<'MSG'
Dataset not found locally.

Expected one of:
  DATASET_DIR=/path/to/dataset
  ../dataset2-master/dataset2-master
  ../dataset-master/dataset-master

Or place the blood-cell dataset manually at:
  data/dataset/images/TRAIN/<class>/*.jpeg
  data/dataset/images/TEST/<class>/*.jpeg
MSG
exit 1
