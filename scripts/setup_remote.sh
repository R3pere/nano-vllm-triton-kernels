#!/usr/bin/env bash
# One-shot environment setup for the P5 benchmark on a fresh Linux GPU machine (written for FunHPC's
# "mini/ubuntu/22.04/3.10" image: Ubuntu 22.04 + Python 3.10, no PyTorch). Safe to run again: every step
# checks whether it is already done and skips it, so after a failure just fix the cause and rerun.
#
#   bash scripts/setup_remote.sh
#
# What it does (log: $P5/setup.log):
#   1. GPU and driver: prints nvidia-smi, stops unless the driver supports CUDA >= 12.8 (torch 2.8 = cu128).
#   2. Disk: everything big goes under $DATA (FunHPC keeps only /data; the system disk is ~30GB).
#   3. System packages: gcc, Python.h, venv, git, tmux (Triton compiles a small C module with gcc and
#      Python.h the first time a model runs; without them it fails only then).
#   4. venv "nano" with the same versions as the dev machine: torch 2.8.0 (cu128), triton 3.4.0,
#      transformers 4.56.2, flash-attn 2.8.3 (prebuilt wheel, checked by sha256), xxhash, numpy.
#   5. venv "vllm" with vLLM $VLLM_VERSION (it brings its own torch, so it must not share the venv).
#   6. Models $TARGET and $DRAFT from HuggingFace (HF_ENDPOINT mirror in China), falling back to ModelScope.
#   7. ShareGPT dataset (scripts/sharegpt.py download).
#   8. Writes p5.env in the repository (paths for scripts/run_p5.sh) and runs a short smoke test of both engines.
#
# Settings (environment variables):
#   DATA=/data          where venvs, models, caches and the dataset go
#   MIRROR=cn           cn: pip from Tsinghua, HuggingFace through hf-mirror.com; none: the default sites
#   PYTHON=python3.10   the interpreter for both venvs
#   TARGET=Qwen3-8B DRAFT=Qwen3-0.6B   models (HuggingFace / ModelScope names under Qwen/)
#   WHEELS=$DATA/wheels  put the flash-attn wheel here by hand if GitHub can't be reached
#   SKIP_VLLM=1, SKIP_SMOKE=1  skip those steps
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:-/data}
P5=$DATA/p5
MIRROR=${MIRROR:-cn}
PYTHON=${PYTHON:-python3.10}
TARGET=${TARGET:-Qwen3-8B}
DRAFT=${DRAFT:-Qwen3-0.6B}
WHEELS=${WHEELS:-$DATA/wheels}
VLLM_VERSION=0.15.1
TORCH_PINS="torch==2.8.0 triton==3.4.0 transformers==4.56.2 xxhash==3.5.0 numpy==2.2.6"
FA_WHEEL=flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
FA_SHA256=75c51f34bb93c5a4438d6b767547ca6c22a9bfc6b29b5646dda98e943cd75d99
FA_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$FA_WHEEL

mkdir -p "$P5" "$WHEELS" "$P5/cache"
exec > >(tee -a "$P5/setup.log") 2>&1
step() { echo; echo "=== $(date '+%F %T') $*"; }
die() { echo "ERROR: $*"; exit 1; }

# caches on the data disk too (pip, HuggingFace, Triton, torch.compile, vLLM)
export PIP_CACHE_DIR=$P5/cache/pip HF_HOME=$P5/cache/hf XDG_CACHE_HOME=$P5/cache TRITON_CACHE_DIR=$P5/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$P5/cache/inductor VLLM_CACHE_ROOT=$P5/cache/vllm UV_CACHE_DIR=$P5/cache/uv
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [ "$MIRROR" = cn ]; then
    export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
    export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
fi
[ -z "${PIP_INDEX_URL:-}" ] || export UV_DEFAULT_INDEX=$PIP_INDEX_URL
export UV_HTTP_TIMEOUT=120
# uv's default of 50 parallel downloads was ~20x slower than 4 in the local rehearsal (servers throttle per client)
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}

step "1. GPU and driver"
nvidia-smi || die "nvidia-smi failed: no GPU driver"
# "CUDA Version: 12.8" (or "CUDA UMD Version" on newer drivers): the newest CUDA the driver supports
cuda=$(nvidia-smi | grep -oE 'CUDA (UMD )?Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+$' | head -1)
echo "driver supports CUDA $cuda"
[ -n "$cuda" ] || die "could not read the CUDA version from nvidia-smi"
python3 -c "import sys; sys.exit(0 if tuple(map(int, '$cuda'.split('.'))) >= (12, 8) else 1)" \
    || die "the driver supports CUDA $cuda, torch 2.8 needs >= 12.8: pick another machine (or switch to the cu126 wheels)"

step "2. Disk"
df -h "$DATA" / || true
free_gb=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
echo "$DATA has ${free_gb}GB free"
[ "$free_gb" -ge 45 ] || [ -f "$P5/p5.done" ] || echo "WARNING: less than 45GB free on $DATA (models ~17GB, two venvs ~15GB, caches); expand the data disk if a step fails"

step "3. System packages"
need=()
command -v gcc >/dev/null || need+=(build-essential)
command -v git >/dev/null || need+=(git)
command -v tmux >/dev/null || need+=(tmux)
command -v "$PYTHON" >/dev/null || need+=(python3.10 python3.10-venv python3.10-dev)
if command -v "$PYTHON" >/dev/null; then
    "$PYTHON" -c "import sysconfig, os, sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')) else 1)" || need+=(python3.10-dev)
    "$PYTHON" -c "import ensurepip, venv" 2>/dev/null || need+=(python3.10-venv)
fi
if [ ${#need[@]} -gt 0 ]; then
    echo "installing: ${need[*]}"
    SUDO=; [ "$(id -u)" = 0 ] || SUDO=sudo
    $SUDO apt-get update && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y "${need[@]}"
fi
"$PYTHON" --version; gcc --version | head -1; git --version

pyver() { "$1" -c "import sys; print('%d.%d' % sys.version_info[:2])"; }
make_venv() {    # make_venv DIR: a venv made from $PYTHON, with uv in it (resolves and downloads much faster than pip)
    [ -x "$1/bin/python" ] || "$PYTHON" -m venv "$1"
    "$1/bin/python" -m pip install -q --upgrade pip uv
}
install() {    # install VENV PACKAGE...: uv pip install into VENV
    local venv=$1; shift
    "$venv/bin/uv" pip install --python "$venv/bin/python" "$@"
}

step "4. venv nano (torch 2.8, flash-attn 2.8.3)"
NANO=$P5/venvs/nano
check_nano() {
    "$NANO/bin/python" - <<'EOF'
import torch, triton, transformers, flash_attn, xxhash, numpy
from flash_attn import flash_attn_varlen_func
assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torch.cuda.is_available(), "torch can't see the GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, "| triton", triton.__version__, "| transformers", transformers.__version__,
      "| flash_attn", flash_attn.__version__, "| numpy", numpy.__version__, "|", torch.cuda.get_device_name())
EOF
}
if [ -x "$NANO/bin/python" ] && check_nano 2>/dev/null; then
    echo "already set up"
else
    make_venv "$NANO"
    [ "$(pyver "$NANO/bin/python")" = 3.10 ] || die "$NANO is not Python 3.10 (the flash-attn wheel is cp310)"
    install "$NANO" $TORCH_PINS
    if [ ! -f "$WHEELS/$FA_WHEEL" ]; then
        # GitHub can be slow or blocked from China: try a public GitHub proxy too; the sha256 check below catches a bad file
        for url in "$FA_URL" "https://ghfast.top/$FA_URL" "https://gh-proxy.com/$FA_URL"; do
            echo "downloading $url"
            if curl -fsSL --retry 2 --connect-timeout 20 -o "$WHEELS/$FA_WHEEL.part" "$url"; then
                mv "$WHEELS/$FA_WHEEL.part" "$WHEELS/$FA_WHEEL"; break
            fi
        done
    fi
    [ -f "$WHEELS/$FA_WHEEL" ] || die "could not download $FA_WHEEL; upload it to $WHEELS/ and rerun"
    echo "$FA_SHA256  $WHEELS/$FA_WHEEL" | sha256sum -c - || { rm -f "$WHEELS/$FA_WHEEL"; die "flash-attn wheel checksum mismatch, deleted it; rerun"; }
    install "$NANO" --no-deps "$WHEELS/$FA_WHEEL"
    install "$NANO" einops    # flash-attn's only dependency besides torch
    check_nano
fi
"$NANO/bin/pip" freeze > "$P5/pip_freeze_nano.txt"

step "5. venv vllm (vLLM $VLLM_VERSION)"
VLLM=$P5/venvs/vllm
check_vllm() {
    "$VLLM/bin/python" -c "import vllm, torch; assert vllm.__version__ == '$VLLM_VERSION', vllm.__version__; assert torch.cuda.is_available(); print('vllm', vllm.__version__, '| torch', torch.__version__, 'cuda', torch.version.cuda)"
}
if [ "${SKIP_VLLM:-0}" = 1 ]; then
    echo "skipped (SKIP_VLLM=1)"
elif [ -x "$VLLM/bin/python" ] && check_vllm 2>/dev/null; then
    echo "already set up"
else
    make_venv "$VLLM"
    install "$VLLM" "vllm==$VLLM_VERSION"
    check_vllm
fi
[ -x "$VLLM/bin/python" ] && "$VLLM/bin/pip" freeze > "$P5/pip_freeze_vllm.txt" || true

step "6. Models ($TARGET, $DRAFT)"
download_model() {    # download_model NAME: Qwen/NAME into $P5/models/NAME, HuggingFace first, then ModelScope
    local name=$1 dir=$P5/models/$1
    if "$NANO/bin/python" "$REPO/scripts/check_model.py" "$dir" 2>/dev/null; then echo "$name: already complete"; return; fi
    echo "$name: from HuggingFace${HF_ENDPOINT:+ ($HF_ENDPOINT)}"
    "$NANO/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/$name', local_dir='$dir', allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*'])" \
        || {
            echo "$name: HuggingFace failed, trying ModelScope"
            install "$NANO" -q modelscope
            "$NANO/bin/python" -c "
from modelscope import snapshot_download
snapshot_download('Qwen/$name', local_dir='$dir', allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*'])"
        }
    "$NANO/bin/python" "$REPO/scripts/check_model.py" "$dir" || die "$name is incomplete in $dir"
}
download_model "$TARGET"
[ "$DRAFT" = "$TARGET" ] || download_model "$DRAFT"

step "7. ShareGPT dataset"
SHAREGPT=$P5/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
if [ -f "$SHAREGPT" ]; then
    echo "already downloaded"
else
    "$NANO/bin/python" "$REPO/scripts/sharegpt.py" download --out "$P5/datasets" \
        || echo "WARNING: ShareGPT download failed. Upload results/p5/sharegpt_sample.json made on another machine instead"
fi

step "8. p5.env and smoke test"
cat > "$REPO/p5.env" <<EOF
# written by scripts/setup_remote.sh $(date '+%F %T'); read by scripts/run_p5.sh
export P5=$P5
export NANO_PY=$NANO/bin/python
export VLLM_PY=$VLLM/bin/python
export MODEL=$P5/models/$TARGET
export DRAFT_MODEL=$P5/models/$DRAFT
export SHAREGPT=$SHAREGPT
export PIP_CACHE_DIR=$PIP_CACHE_DIR HF_HOME=$HF_HOME XDG_CACHE_HOME=$XDG_CACHE_HOME TRITON_CACHE_DIR=$TRITON_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT
EOF
cat "$REPO/p5.env"
if [ "${SKIP_SMOKE:-0}" != 1 ]; then
    cd "$REPO"
    smoke="--model $P5/models/$TARGET --workload random --num-seqs 8 --input-len 32 64 --output-len 16 32 --max-num-seqs 16 --gpu-memory-utilization 0.5"
    echo "--- nano-vllm (CUDA graph, n-gram)"; "$NANO/bin/python" bench.py $smoke --spec ngram
    if [ "${SKIP_VLLM:-0}" != 1 ]; then echo "--- vLLM"; "$VLLM/bin/python" bench.py $smoke --engine vllm; fi
fi
touch "$P5/p5.done"
step "setup finished. Next: bash scripts/run_p5.sh"
