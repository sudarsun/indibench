#!/usr/bin/env bash
# Provision a fresh Ubuntu 24.04 / 26.04 machine for indibench.
# Target GPUs (Turing or newer):
#   workstation: RTX 2080 Ti, RTX 3090, RTX 4090
#   data center: L40S, A100 (40/80 GB), H100 (PCIe/SXM/NVL), H200 (SXM/NVL)
#
#   bash scripts/setup_ubuntu.sh                 # driver from Ubuntu repo + .venv with hf/quant extras
#   bash scripts/setup_ubuntu.sh --vllm          # also create .venv-vllm
#   bash scripts/setup_ubuntu.sh --driver nvidia # driver from NVIDIA's CUDA apt repo instead
#   bash scripts/setup_ubuntu.sh --driver skip   # driver already installed (e.g. second run after reboot)
#   bash scripts/setup_ubuntu.sh --fabric-manager yes   # force Fabric Manager (8-GPU HGX with NVSwitch)
#
# Data-center GPUs are detected automatically and get the -server driver branch, persistence
# daemon, and (on NVSwitch/HGX baseboards) a version-matched, held Fabric Manager.
# The script is idempotent: re-run it after the reboot that the driver install requires.
set -euo pipefail

DRIVER="ubuntu"          # ubuntu | nvidia | skip
MIN_DRIVER=580           # CUDA 13.x (PyTorch >= 2.11 default wheels) needs driver >= 580
CUDA_TOOLKIT=0
WITH_VLLM=0
FLASH_ATTN=auto          # auto | yes | no   (auto = yes on Ampere/Ada/Hopper, no on Turing)
PY_VER="3.12"
VENV=".venv"
VENV_VLLM=".venv-vllm"
FABRIC_MANAGER=auto      # auto | yes | no
YES=0

usage() { sed -n '2,17p' "$0"; exit 0; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --driver) DRIVER="$2"; shift 2 ;;
    --cuda-toolkit) CUDA_TOOLKIT=1; shift ;;
    --vllm) WITH_VLLM=1; shift ;;
    --flash-attn) FLASH_ATTN="$2"; shift 2 ;;
    --fabric-manager) FABRIC_MANAGER="$2"; shift 2 ;;
    --python) PY_VER="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    -y|--yes) YES=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1"; usage ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*"; exit 1; }
APT_Y=(-y); [[ $YES -eq 1 ]] && export DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------ 0. OS + GPU checks
. /etc/os-release
log "Detected $PRETTY_NAME"
case "$VERSION_ID" in
  24.04|26.04) ;;
  *) warn "Tested on Ubuntu 24.04 and 26.04 only; continuing anyway." ;;
esac
CUDA_REPO="ubuntu${VERSION_ID/./}"
ARCH="$(dpkg --print-architecture)"; [[ "$ARCH" == "amd64" ]] && NV_ARCH="x86_64" || NV_ARCH="sbsa"

if ! lspci 2>/dev/null | grep -qi nvidia; then
  sudo apt-get update && sudo apt-get install "${APT_Y[@]}" pciutils
fi
lspci | grep -i 'vga\|3d' | grep -i nvidia || warn "No NVIDIA GPU visible on the PCI bus."

# Data-center GPU?  -> server driver branch + persistence daemon (+ Fabric Manager on NVSwitch boards)
DC_REGEX='A100|A800|H100|H800|H200|GH100|GH200|GA100|L40S|L40|AD102GL'
DATACENTER=0
if lspci | grep -i nvidia | grep -qE "$DC_REGEX"; then DATACENTER=1; fi
N_GPUS="$(lspci | grep -i nvidia | grep -ciE '3d controller|vga' || true)"
if [[ "$FABRIC_MANAGER" == "auto" ]]; then
  # NVSwitch shows up as a separate NVIDIA PCI bridge on 8-GPU HGX A100/H100/H200 baseboards.
  # 4-GPU HGX boards (direct NVLink mesh) and PCIe/NVL cards do NOT need Fabric Manager.
  if lspci | grep -i nvidia | grep -qi nvswitch; then FABRIC_MANAGER=yes; else FABRIC_MANAGER=no; fi
fi
if (( DATACENTER )); then
  log "Data-center GPU(s) detected ($N_GPUS GPUs); Fabric Manager: $FABRIC_MANAGER"
fi

# ------------------------------------------------------------------ 1. base packages
log "Installing base packages"
sudo apt-get update
sudo apt-get install "${APT_Y[@]}" \
  build-essential git curl wget ca-certificates pkg-config gnupg \
  "linux-headers-$(uname -r)" dkms ubuntu-drivers-common \
  python3-venv python3-dev \
  fonts-noto-core libraqm0 \
  htop nvtop tmux

# ------------------------------------------------------------------ 2. NVIDIA driver
driver_version() { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1; }
driver_ok() {
  local v; v="$(driver_version)" || return 1
  [[ -n "$v" ]] && (( ${v%%.*} >= MIN_DRIVER ))
}

highest_branch() {  # $1 = apt package-name regex with one ([0-9]+) group
  apt-cache search --names-only "$1" | sed -E "s/${1//\$/}.*/\\1/" | grep -E '^[0-9]+$' \
    | sort -n | awk -v m="$MIN_DRIVER" '$1>=m' | tail -1
}

install_driver_ubuntu() {
  # Ubuntu ships pre-built, signed kernel modules for these branches, so this path also works
  # with Secure Boot enabled. Data-center cards use the -server branch: Ubuntu's
  # nvidia-fabricmanager-<br> / libnvidia-nscq-<br> packages are built to match it.
  local br
  if (( DATACENTER )); then
    br="$(highest_branch '^nvidia-driver-([0-9]+)-server$')"
  else
    br="$(highest_branch '^nvidia-driver-([0-9]+)-open$')"
  fi
  if [[ -z "$br" ]]; then
    warn "Ubuntu archive has no suitable driver branch >= $MIN_DRIVER; switching to NVIDIA's repository."
    install_driver_nvidia; return
  fi
  if (( DATACENTER )); then
    log "Installing nvidia-driver-${br}-server (Ubuntu archive, data-center branch)"
    sudo ubuntu-drivers install --gpgpu "nvidia:${br}-server"
    sudo apt-get install "${APT_Y[@]}" "nvidia-utils-${br}-server" || true
    if [[ "$FABRIC_MANAGER" == "yes" ]]; then
      log "Installing Fabric Manager + NSCQ for branch $br (must match the driver version exactly)"
      sudo apt-get install "${APT_Y[@]}" "nvidia-fabricmanager-${br}" "libnvidia-nscq-${br}"
      # Hold the matched set so unattended upgrades cannot split driver and Fabric Manager versions.
      sudo apt-mark hold "nvidia-driver-${br}-server" "nvidia-fabricmanager-${br}" "libnvidia-nscq-${br}"
      sudo systemctl enable nvidia-fabricmanager
    fi
  else
    log "Installing nvidia-driver-${br}-open (Ubuntu archive, open kernel modules: Turing and newer)"
    sudo ubuntu-drivers install "nvidia:${br}-open"
  fi
}

add_nvidia_repo() {
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    log "Adding NVIDIA CUDA apt repository ($CUDA_REPO/$NV_ARCH)"
    local tmp; tmp="$(mktemp -d)"
    wget -q -O "$tmp/cuda-keyring.deb" \
      "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/${NV_ARCH}/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i "$tmp/cuda-keyring.deb"
    rm -rf "$tmp"
    sudo apt-get update
  fi
}

install_driver_nvidia() {
  add_nvidia_repo
  if dpkg -l | grep -qE '^ii\s+nvidia-driver-[0-9]+'; then
    warn "Ubuntu-archive NVIDIA packages are installed; they conflict with NVIDIA's repo packages."
    warn "Remove them first:  sudo apt purge 'nvidia-driver-*' 'linux-modules-nvidia-*' 'nvidia-fabricmanager-*' && sudo apt autoremove"
    die "aborting to avoid a mixed driver install"
  fi
  # Pin one branch so driver and Fabric Manager can never drift apart.
  local br
  br="$(highest_branch '^nvidia-driver-pinning-([0-9]+)$')"
  if [[ -n "$br" ]]; then
    log "Pinning NVIDIA driver branch $br"
    sudo apt-get install "${APT_Y[@]}" "nvidia-driver-pinning-${br}"
  fi
  log "Installing nvidia-open (NVIDIA repo, DKMS-built open kernel modules)"
  sudo apt-get install "${APT_Y[@]}" nvidia-open
  if [[ "$FABRIC_MANAGER" == "yes" ]]; then
    [[ -n "$br" ]] || die "Fabric Manager needs a pinned branch; install nvidia-driver-pinning-<branch> first"
    log "Installing Fabric Manager for branch $br"
    sudo apt-get install "${APT_Y[@]}" "cuda-drivers-fabricmanager-${br}"
    sudo systemctl enable nvidia-fabricmanager
  fi
  if mokutil --sb-state 2>/dev/null | grep -qi enabled; then
    warn "Secure Boot is ON: DKMS modules must be signed. You will be asked for a MOK password now or"
    warn "at next boot choose 'Enroll MOK' in the blue MOK manager screen and enter that password."
  fi
}

case "$DRIVER" in
  skip) log "Skipping driver install" ;;
  ubuntu|nvidia)
    if driver_ok; then
      log "Driver $(driver_version) already installed (>= $MIN_DRIVER) - nothing to do"
    else
      if [[ -n "$(driver_version || true)" ]]; then
        warn "Driver $(driver_version) is older than $MIN_DRIVER; upgrading."
      fi
      if [[ "$DRIVER" == "ubuntu" ]]; then install_driver_ubuntu; else install_driver_nvidia; fi
      echo
      warn "Driver installed. REBOOT now, then re-run:  bash scripts/setup_ubuntu.sh --driver skip $( ((WITH_VLLM)) && echo --vllm )"
      exit 0
    fi ;;
  *) die "--driver must be ubuntu|nvidia|skip" ;;
esac

if ! driver_ok; then
  die "nvidia-smi does not report a driver >= $MIN_DRIVER. Reboot if you just installed it; see docs/SETUP_UBUNTU.md (Troubleshooting)."
fi
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv
if (( DATACENTER )); then
  sudo systemctl enable --now nvidia-persistenced 2>/dev/null || sudo nvidia-smi -pm 1 || true
  if [[ "$FABRIC_MANAGER" == "yes" ]]; then
    sudo systemctl start nvidia-fabricmanager || true
    if ! systemctl is-active --quiet nvidia-fabricmanager; then
      die "nvidia-fabricmanager is not running: CUDA will fail with 'system not yet initialized'. Check
  journalctl -u nvidia-fabricmanager  (usual cause: Fabric Manager version != driver version)."
    fi
    log "Fabric Manager active"
  fi
  if nvidia-smi -q | grep -qiE 'MIG Mode[^\n]*Current *: *Enabled'; then
    warn "MIG is enabled: each process sees one slice. Select it with CUDA_VISIBLE_DEVICES=MIG-<uuid> (nvidia-smi -L)."
  fi
fi
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
CC_MAJOR="${CC%%.*}"
(( CC_MAJOR >= 7 )) || die "GPU compute capability $CC is below Turing (7.5); CUDA 13 PyTorch wheels do not support it."

# ------------------------------------------------------------------ 3. optional CUDA toolkit (nvcc)
# NOT needed to run indibench: PyTorch/vLLM wheels bundle the CUDA runtime libraries.
# Only needed to compile extensions (flash-attn without a matching prebuilt wheel, custom kernels).
if [[ $CUDA_TOOLKIT -eq 1 ]]; then
  add_nvidia_repo
  log "Installing CUDA toolkit (nvcc) from NVIDIA's repository"
  sudo apt-get install "${APT_Y[@]}" cuda-toolkit
  grep -q '/usr/local/cuda/bin' ~/.bashrc || {
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
  }
  export PATH=/usr/local/cuda/bin:$PATH
  nvcc --version | tail -1
fi

# ------------------------------------------------------------------ 4. uv + Python
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv (Python package/venv manager)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
# Python 3.12 has the widest prebuilt-wheel coverage (bitsandbytes, flash-attn, vLLM).
# Ubuntu 26.04's system Python is 3.14 and 24.04's is 3.12; uv fetches a managed 3.12 either way.
log "Creating $VENV with Python $PY_VER"
uv venv --python "$PY_VER" "$VENV" --allow-existing
PY="$REPO_DIR/$VENV/bin/python"

# ------------------------------------------------------------------ 5. PyTorch + indibench
log "Installing PyTorch (CUDA 13 wheels from PyPI) and indibench[hf,quant,dev]"
uv pip install --python "$PY" torch torchvision
uv pip install --python "$PY" -e ".[hf,quant,api,dev,math]" fonttools

if [[ "$FLASH_ATTN" == "auto" ]]; then
  if (( CC_MAJOR >= 8 )); then FLASH_ATTN=yes; else FLASH_ATTN=no; fi
fi
if [[ "$FLASH_ATTN" == "yes" ]]; then
  log "Installing flash-attn (Ampere/Ada/Hopper only; may compile for 10-60 min if no wheel matches)"
  MAX_JOBS="${MAX_JOBS:-4}" uv pip install --python "$PY" flash-attn --no-build-isolation \
    || warn "flash-attn install failed - indibench falls back to PyTorch SDPA automatically (slightly slower)."
else
  log "Skipping flash-attn (GPU compute capability $CC; FlashAttention-2 requires >= 8.0). SDPA will be used."
fi

# ------------------------------------------------------------------ 6. optional vLLM venv
# vLLM pins its own torch version, so it lives in a separate venv to avoid conflicts.
if [[ $WITH_VLLM -eq 1 ]]; then
  log "Creating $VENV_VLLM with vLLM"
  uv venv --python "$PY_VER" "$VENV_VLLM" --allow-existing
  uv pip install --python "$VENV_VLLM/bin/python" vllm --torch-backend=auto
  uv pip install --python "$VENV_VLLM/bin/python" -e ".[api]"
  if (( CC_MAJOR < 8 )); then
    warn "Turing GPU: vLLM runs in fp16 with fallback attention kernels; some models may not start. Use --backend hf if so."
  fi
fi

# ------------------------------------------------------------------ 7. verification
log "Verifying"
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA runtime", torch.version.cuda, "| available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch cannot see the GPU"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i}: {p.name} sm_{p.major}{p.minor} {p.total_memory/1024**3:.1f} GB")
    # sm_89 (RTX 4090 / L40S) runs sm_86 binaries: same major, lower-or-equal minor is compatible.
    archs = [a.split("_")[1] for a in torch.cuda.get_arch_list() if a.startswith(("sm_", "compute_"))]
    ok = any(int(a[:-1]) == p.major and int(a[-1]) <= p.minor for a in archs if a.isdigit())
    assert ok, f"this torch build has no kernels for sm_{p.major}{p.minor}: {torch.cuda.get_arch_list()}"
dt = torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16
x = torch.randn(1024, 1024, device="cuda", dtype=dt)
print("matmul ok:", float((x @ x).float().abs().mean()) > 0, "dtype", dt)
EOF
"$VENV/bin/indibench" env | tail -12
"$VENV/bin/python" -m pytest -q tests 2>&1 | tail -2

cat <<EOF

Done. Next steps:
  source $VENV/bin/activate
  hf auth login                          # or: export HF_TOKEN=hf_xxx  (MILU, FLORES+, Belebele, IN22 are gated)
  export HF_HOME=/path/to/big/disk/hf    # optional: model + dataset cache (~15-60 GB)
  indibench check tamil_open
  indibench run -c configs/$( ((CC_MAJOR < 8)) && echo rtx2080ti_hf.yaml || echo rtx3090_hf.yaml ) -n 20
EOF
