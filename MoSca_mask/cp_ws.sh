#!/usr/bin/env bash
set -euo pipefail

# # The image in custom_mosca_trained is resized but the aspcet ratio is not preserved.
# # Copy images from /data/dataset/custom_mosca_trained/{seq_name}/images
# # to /data/dataset/custom_mosca_trained_mask/{seq_name}/images

# SRC_BASE="/data/dataset/custom_mosca_trained"
# DST_BASE="/data/dataset/custom_mosca_trained_mask"

# # If seq names are provided as args, use them; otherwise copy for all sequences found in SRC_BASE
# if [ "$#" -gt 0 ]; then
#   SEQS=("$@")
# else
#   # List immediate subdirectories of SRC_BASE as sequence names
#   IFS=$'\n' read -r -d '' -a SEQS < <(find "$SRC_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort && printf '\0')
# fi

# for SEQ in "${SEQS[@]}"; do
#   SRC_IMAGES="$SRC_BASE/$SEQ/images"
#   DST_IMAGES="$DST_BASE/$SEQ/images"

#   if [ ! -d "$SRC_IMAGES" ]; then
#     echo "[skip] $SEQ: source images directory not found: $SRC_IMAGES" >&2
#     continue
#   fi

#   mkdir -p "$DST_IMAGES"
#   # Copy preserving attributes; copy contents of images into destination images
#   cp -a "$SRC_IMAGES"/. "$DST_IMAGES"/
#   echo "[ok] Copied images for $SEQ"
# done

# echo "Done."

# Use som initialized code.
SRC_BASE="/data/dataset/custom_som_initialized/images"
# DST_BASE="/data/dataset/custom_mosca_trained_masked"

# If seq names are provided as args, use them; otherwise copy for all sequences found in SRC_BASE
if [ "$#" -gt 0 ]; then
  SEQS=("$@")
else
  # List immediate subdirectories of SRC_BASE as sequence names
  IFS=$'\n' read -r -d '' -a SEQS < <(find "$SRC_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort && printf '\0')
fi

# copy from custom_som_initialized to custom_mosca_trained_mask
echo "Copying images..."

IMG_SRC_BASE="/data/dataset/custom_som_initialized/images"
IMG_DST_BASE="/data/dataset/custom_mosca_trained_masked"

for SEQ in "${SEQS[@]}"; do
  IMG_SRC="$IMG_SRC_BASE/$SEQ"
  IMG_DST="$IMG_DST_BASE/$SEQ/images"
  
  
  # Copy images
  mkdir -p "$IMG_DST"
  cp -a "$IMG_SRC"/. "$IMG_DST"/
  echo "[ok] Copied images for $SEQ"
done

echo "Images copying done."

# copy mask 
# copy from source /data/dataset/custom_som_initialized/masks/{seq_name} to dest /data/dataset/custom_mosca_trained_mask/{seq_name}/mask
# if dest does not have the seq_name, ignore it, print skip
# if dest has the seq_name but source does not have the seq_name, print missing warning

echo "Copying masks..."

MASK_SRC_BASE="/data/dataset/custom_som_initialized/masks"
MASK_DST_BASE="/data/dataset/custom_mosca_trained_masked"

for SEQ in "${SEQS[@]}"; do
  MASK_SRC="$MASK_SRC_BASE/$SEQ"
  MASK_DST="$MASK_DST_BASE/$SEQ/mask"
  
  # Copy masks
  mkdir -p "$MASK_DST"
  cp -a "$MASK_SRC"/. "$MASK_DST"/
  echo "[ok] Copied masks for $SEQ"
done

echo "Mask copying done."
