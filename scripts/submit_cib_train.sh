#!/bin/bash
#SBATCH -A mp107c
#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 32
#SBATCH -J cib_train
#SBATCH -o runs/sbatch_cib_train_%j.out
#SBATCH -e runs/sbatch_cib_train_%j.out
# ---------------------------------------------------------------------------
# Joint 3-channel CIB flow-matching + multi-channel WPH training (SPT-3G 95/150/220).
# Measured on A100: ~2.74 s/train-step, ~9 GB peak (fits 40 GB). 50 epochs ~= 12 h.
# Self-contained: rebuilds any missing artifact (dataset / prior / targets), then
# trains -> samples -> plots. Submit with:  sbatch scripts/submit_cib_train.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# --- config knobs ---
REPO=/global/cfs/cdirs/mp107c/yomori/repo/AgoraSynth
PY=/global/cfs/cdirs/des/yomori/packages_perl/miniconda3/envs/jaxkdk2/bin/python
EPOCHS=50
BATCH=32
LR=2e-4
CHANNELS="64 128 256"
DATA=data/train_cib.npz
PRIOR=runs/wph_prior_cib.npz
TARGETS=data/wph_targets_cib.npz
CKPT=checkpoints/fm_wph_cib.pkl

cd "$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # 9 GB peak; no need to grab the whole card
echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  start=$(date)"
$PY -c "import jax; print('jax devices:', jax.devices())"

# --- 1. dataset (10000 co-located ZEA patches, 3 bands) ---
if [[ ! -f "$DATA" ]]; then
  echo "[dataset] building $DATA ..."
  JAX_PLATFORMS=cpu $PY scripts/build_dataset_cib.py \
      --n-patches 10000 --patch-size-deg 5 --pixel-size-arcmin 1.6 --out "$DATA"
else
  echo "[dataset] reuse $DATA"
fi

# --- 2. WPH prior (persample mode uses only its CONFIG, not mean/cov -> cheap, no D4) ---
if [[ ! -f "$PRIOR" ]]; then
  echo "[prior] building $PRIOR ..."
  $PY scripts/build_wph_prior_cib.py --data "$DATA" \
      --n-patches 1000 --J 6 --L 4 --A 4 --dn 0 --out "$PRIOR"
else
  echo "[prior] reuse $PRIOR"
fi

# --- 3. per-sample WPH targets (one feature vector per training patch) ---
# Use the SHARDED builder: the single-process precompute dies deterministically
# after ~470 batches (~patch 3700) from a per-process JAX/XLA buildup. Sharding
# runs each block in a fresh subprocess (resumable) and concatenates.
if [[ ! -f "$TARGETS" ]]; then
  echo "[targets] building $TARGETS (sharded) ..."
  $PY scripts/build_wph_targets_cib_sharded.py --data "$DATA" \
      --wph-prior "$PRIOR" --out "$TARGETS" --shard 2000
else
  echo "[targets] reuse $TARGETS"
fi

# --- 4. train joint 3-channel FM + multi-channel WPH (persample) ---
echo "[train] $EPOCHS epochs ..."
$PY scripts/train_fm_wph_cib.py \
    --data "$DATA" --wph-prior "$PRIOR" --wph-targets "$TARGETS" \
    --wph-mode persample --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
    --lambda-wph 1.0 --wph-t-min 0.5 --wph-warmup-epochs 5 --wph-chunk-size 1 \
    --channels $CHANNELS --save-every 5 --out "$CKPT"

# --- 5. sample + diagnostics + realization grid ---
echo "[sample] ..."
$PY scripts/sample_fm_cib.py --ckpt "$CKPT" --n-samples 16 --n-steps 30 \
    --diagnostics --out samples/fm_cib.npz
$PY scripts/plot_cib_realizations.py --samples samples/fm_cib.npz \
    --data "$DATA" --out samples/cib_realizations.png

echo "done=$(date)  ckpt=$CKPT"
