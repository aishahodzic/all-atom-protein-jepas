# Jarvis: locked La-Proteina StructTokenBench evaluation

The named protocol is additive. Invocations without `--protocol` retain the
existing integration. `upstream-l600-v1` locks the publication run to:

- online/student encoder;
- StructTokenBench's selected-residue `filter_length=600`;
- no La-Proteina checkpoint-training cohort filters;
- upstream batch-mean classification metrics and upstream-effective class
  weighting (`0.5` for the present class in a single-class minibatch);
- the upstream probe schedule: seed 1234, learning rate 0.001, and a 10,000-step
  ceiling; our intentional runtime changes are validation every second epoch,
  patience 10 early stopping (about 20 unimproved epochs), and checkpointing at
  every validation (the smoke has its separately locked one-epoch/two-batch limits);
- content-addressed checkpoint/token caches and an immutable result manifest.

It does **not** cap the full chain at 600. Homo domains can satisfy STB's L600
rule while the chain encoded before domain selection is as long as 1023.

## 1. Stage the BindInt smoke data

First sync this exact working tree to Jarvis. The protocol files are currently
local workspace changes; a plain clone of an older commit will not contain
them. Commit/push/pull them, or copy the working tree explicitly.

Then copy this prepared archive to Jarvis:

```text
artifacts/hf/stb-bindint-upstream-L600-v1.tar.zst
```

Expected SHA-256:

```text
2bc385fb13d271c485ccfb98ede8a545f8da14e895c4c05018d6499bb18f9b75
```

Then, on Jarvis:

```bash
mkdir -p /workspace/stb-bindint
echo "2bc385fb13d271c485ccfb98ede8a545f8da14e895c4c05018d6499bb18f9b75  /workspace/stb-bindint-upstream-L600-v1.tar.zst" \
  | sha256sum -c -
command -v zstd >/dev/null || {
  echo "Install the zstd command before extracting" >&2
  exit 1
}
tar --zstd -xf /workspace/stb-bindint-upstream-L600-v1.tar.zst \
  -C /workspace/stb-bindint

export DATA_PATH=/workspace/stb-bindint
export PDB_DATA_ROOT="$DATA_PATH/struct_token_bench_pdb_data"
export STB_RELEASE_DATA_ROOT="$DATA_PATH/struct_token_bench_release_data"
```

This minimal archive covers BindInt only. Flex and Homo require their own task
data/CIF closure or the full corpus; do not interpret a missing-file failure on
those tasks as an encoder failure.

The gitignored checkpoint files are not supplied by a repository clone. For
the requested `jepa-rand-mask-50` smoke, stage both files at these exact paths:

```text
StructTokenBench/100k-checkpoints/jepa/random_masking_50pct/100k.ckpt
StructTokenBench/100k-checkpoints/shared/exp_config_jepa_nn130m.json
```

Verify them before launching:

```bash
cd /workspace/la-proteina/StructTokenBench
echo "bbc0761375db3813c981e4d09379b57b314085db2c412d06bf0ace2e56523896  100k-checkpoints/jepa/random_masking_50pct/100k.ckpt" | sha256sum -c -
echo "98e0350d418e7d840fa245ddfd7150064e87a9ac74c72c9150a946f0f66ccd9d  100k-checkpoints/shared/exp_config_jepa_nn130m.json" | sha256sum -c -
```

If you transfer from the current cluster directly, an adapted command such as
`rsync -ah --progress <source> <jarvis-host>:/workspace/...` is sufficient;
replace the placeholders with your Jarvis SSH endpoint. Do not copy old token
caches—the locked run creates content-addressed caches itself.

## 2. Run the full-cohort smoke

From the StructTokenBench directory in the cloned repository:

```bash
cd /workspace/la-proteina/StructTokenBench

export CKPT_ROOT="$PWD/100k-checkpoints"
export EVALUATION_RUN_ID="jarvis-$(date -u +%Y%m%dT%H%M%SZ)-p$$"
export STB_LOG_ROOT="/workspace/stb-logs/runs/$EVALUATION_RUN_ID"
export CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
export STB_SKIP_DATA_SETUP=1
export STB_SMOKE=1

bash src/script/run_eval.sh \
  --protocol=upstream-l600-v1 \
  --checkpoint=100k \
  --experiment=jepa-rand-mask-50 \
  --tasks=BindInt
```

The first invocation can bootstrap `pstbench`, but installing the full protein
stack is substantially longer than creating the Conda environment itself. For
the deadline, prefer an already-tested GPU environment and use
`STB_SKIP_BOOTSTRAP=1 PYTHON_BIN=python`.

`STB_SMOKE=1` still preprocesses and fingerprints every cohort row. It only
limits downstream probe training/validation to two batches. Expected BindInt
row counts are:

| Split | Rows |
|---|---:|
| train | 1353 |
| validation | 256 |
| fold_test | 671 |
| superfamily_test | 273 |

The run must abort, rather than reduce these counts, if chain resolution,
parsing, or encoding fails. STB's existing residue-alignment behavior remains
intact: in BindInt fold-test row `2fmd_A`, two coordinate-absent `X` positions
(both negative labels) are omitted and recorded as
`alignment_dropped_label_positions=2`; no protein row is discarded.

## 3. Run the full probe

After the smoke succeeds, remove the smoke flag:

```bash
unset STB_SMOKE

# Give the full probe its own immutable output namespace. This also makes
# concurrent experiments safe: each invocation gets its own summary CSV,
# result files, trainer directory, and manifest directory.
export EVALUATION_RUN_ID="jarvis-$(date -u +%Y%m%dT%H%M%SZ)-p$$-full"
export STB_LOG_ROOT="/workspace/stb-logs/runs/$EVALUATION_RUN_ID"

bash src/script/run_eval.sh \
  --protocol=upstream-l600-v1 \
  --checkpoint=100k \
  --experiment=jepa-rand-mask-50 \
  --tasks=BindInt
```

Authoritative manifests are written under:

```text
$STB_LOG_ROOT/manifests/<run-id>/
```

Each includes checkpoint/config/source hashes, resolved probe settings, exact
cohort counts and hashes, residue-alignment drop counts, probe metrics/losses,
and runtime provenance.
The existing text results and summary CSV remain convenience artifacts.

Do not launch multiple encoders concurrently against one cold extraction. The
native La-Proteina graph cache is shared; warm it with one smoke first, or give
each concurrent run a separate writable extraction.
