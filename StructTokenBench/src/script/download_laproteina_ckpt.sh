#!/bin/bash
#bash ~/projects/StructTokenBench/src/script/download_laproteina_ckpt.sh [ca-cond-baseline|fully_latent|jepa-random-masking|all]

download_if_missing () {
  local filename=$1
  local dest=$2
  if [ ! -f "$dest" ]; then
    mkdir -p "$(dirname "$dest")"
    TMP_DIR=$(mktemp -d)
    hf download msc-thesis-org/checkpoints "$filename" --repo-type dataset --token "$HF_TOKEN" --local-dir "$TMP_DIR"
    mv "$TMP_DIR/$filename" "$dest"
    rm -rf "$TMP_DIR"
  fi
}

download_ca_cond_baseline () {
  download_if_missing "ae/ca-cond/100k-EMA.ckpt" \
    "StructTokenBench/src/checkpoints/laproteina/ae/ca-cond/100k-EMA.ckpt"
}

download_fully_latent () {
  download_if_missing "ae/fully_latent/last-EMA.ckpt" \
    "StructTokenBench/src/checkpoints/laproteina/ae/fully_latent/last-EMA.ckpt"
}

download_jepa_random_masking () {
  download_if_missing "jepa/random_masking/last.ckpt" \
    "StructTokenBench/src/checkpoints/laproteina/jepa/random_masking/last.ckpt"

  # TODO: confirm the exact exp_config filename in the HF repo listing for this run
  # (locally it's named exp_config_jepa_random_mask_with_lr.json)
  download_if_missing "jepa/random_masking/exp_config_random_masking.json" \
    "StructTokenBench/src/checkpoints/laproteina/jepa/random_masking/exp_config_random_masking.json"
}

EXPERIMENT=$1

if [ -z "$EXPERIMENT" ]; then
  echo "Which experiment's checkpoints do you want to download?"
  select choice in "ca-cond-baseline" "fully_latent" "jepa-random-masking" "all"; do
    if [ -n "$choice" ]; then
      EXPERIMENT=$choice
      break
    fi
    echo "Invalid choice, try again."
  done
fi

case "$EXPERIMENT" in
  ca-cond-baseline)
    download_ca_cond_baseline
    ;;
  fully_latent)
    download_fully_latent
    ;;
  jepa-random-masking)
    download_jepa_random_masking
    ;;
  all)
    download_ca_cond_baseline
    download_fully_latent
    download_jepa_random_masking
    ;;
  *)
    echo "Unknown experiment: $EXPERIMENT" >&2
    echo "Expected one of: ca-cond-baseline, fully_latent, jepa-random-masking, all" >&2
    exit 1
    ;;
esac
