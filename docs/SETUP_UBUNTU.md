# Setting up a fresh Ubuntu 24.04 / 26.04 machine for indibench

This guide covers the NVIDIA driver, CUDA, the Python environment and indibench itself, for every
supported GPU:

| class | GPUs |
|---|---|
| **Workstation / consumer** | RTX 2080 Ti, RTX 3090, RTX 4090 |
| **Data center** | L40S, A100 (40/80 GB, PCIe/SXM), H100 (PCIe/SXM/NVL), H200 (SXM/NVL) |

Any Turing-or-newer NVIDIA GPU should work. Older ones do not; see §0.

To do it all in one go, run the setup script. It performs the steps below, can be re-run safely,
and stops once for the reboot that a new driver needs:

```bash
git clone <your copy of indibench> && cd indibench      # or unzip indibench.zip
bash scripts/setup_ubuntu.sh            # installs the driver, then asks you to reboot
sudo reboot
bash scripts/setup_ubuntu.sh --driver skip          # add --vllm for a vLLM environment
```

The script detects data-center GPUs on its own. For them it installs the **server** driver branch,
enables the persistence daemon, and on 8-GPU HGX/NVSwitch baseboards installs and holds a **Fabric
Manager** that matches the driver version.

| option | effect |
|---|---|
| `--driver ubuntu\|nvidia\|skip` | Where the driver comes from. `ubuntu` is the default and works with Secure Boot. |
| `--fabric-manager auto\|yes\|no` | `auto` installs Fabric Manager only when NVSwitch devices are present. |
| `--cuda-toolkit` | Also install `nvcc`. Only needed to compile extensions. |
| `--vllm` | Create a second environment, `.venv-vllm`. |
| `--flash-attn yes\|no\|auto` | Install FlashAttention-2. `auto` means yes on Ampere and newer, no on the 2080 Ti. |
| `--python 3.12` | Python version for the virtual environments. |

---

## 0. What you need and why

| component | version | notes |
|---|---|---|
| NVIDIA driver | **≥ 580** (595+ recommended) | PyTorch's default CUDA wheels are CUDA 13.x, which requires driver ≥ 580. |
| CUDA runtime | bundled in the PyTorch/vLLM wheels | A system CUDA toolkit is **not** required to run indibench. |
| CUDA toolkit (`nvcc`) | optional | Only needed for compiling flash-attn or custom kernels. |
| Python | 3.12 in a venv managed by `uv` | Ubuntu 26.04's system Python is **3.14** and 24.04's is **3.12**. Using 3.12 everywhere gives the widest choice of prebuilt wheels. |
| GPU architecture | Turing (sm_75) or newer | PyTorch 2.15's CUDA 13 wheels drop Maxwell, Pascal and Volta and keep Turing. GTX 10-series and V100 are out. |

### GPU capabilities and what indibench does with them

| GPU | arch / CC | VRAM | bf16 | FlashAttn-2 | FP8 | indibench profile | typical model on one GPU |
|---|---|---|---|---|---|---|---|
| RTX 2080 Ti | Turing 7.5 | 11 GB | **no**, so fp16 | **no**, so SDPA | no | `rtx2080ti` | 2–4B in fp16; 7–8B in 4-bit |
| RTX 3090 | Ampere 8.6 | 24 GB | yes | yes | no | `rtx3090` | 7–8B in bf16; 12–14B in 8-bit |
| RTX 4090 | Ada 8.9 | 24 GB | yes | yes | **yes** | `rtx4090` | 7–8B in bf16; 12–14B in FP8 |
| L40S | Ada 8.9 | 48 GB | yes | yes | **yes** | `l40s` | up to 14B in bf16; 32B in 4-bit/AWQ |
| A100 40 GB | Ampere 8.0 | 40 GB | yes | yes | no | `a100-40g` | up to 14B in bf16; 32B in 4-bit |
| A100 80 GB | Ampere 8.0 | 80 GB | yes | yes | no | `a100-80g` | up to 32B in bf16; 70B+ in 4-bit/AWQ |
| H100 SXM/PCIe | Hopper 9.0 | 80 GB | yes | yes (FA3 in vLLM) | **yes** | `h100` | up to 32B in bf16; 70B+ in AWQ |
| H100 NVL | Hopper 9.0 | 94 GB | yes | yes (FA3 in vLLM) | **yes** | `h100` | as H100, plus 72B in FP8 |
| H200 | Hopper 9.0 | 141 GB | yes | yes (FA3 in vLLM) | **yes** | `h200` | up to 32–38B in bf16 with long context; 72B in FP8 |

What indibench does at start-up:

* It detects the GPU, applies the matching profile (batch size, context length, per-image pixel
  budget, vLLM memory share) and checks whether your model will fit.
* On the 2080 Ti it switches bf16 to fp16 and FlashAttention-2 to SDPA. It warns about models that
  overflow in fp16 (Gemma) and about FP8 on cards without FP8 tensor cores (2080 Ti, 3090, A100).
* A **MIG slice** of an A100/H100 is sized by the memory it actually exposes, not by the full
  card's memory.
* Everything it changed is recorded under `hardware_notes` in `results.json` and `report.md`.

`indibench env -m <model> [--quantization 4bit|8bit|fp8]` prints all of this before you start a run.

---

## 1. Base packages

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y build-essential git curl wget ca-certificates pkg-config gnupg \
    linux-headers-$(uname -r) dkms ubuntu-drivers-common python3-venv python3-dev \
    fonts-noto-core libraqm0 htop nvtop tmux
```

`fonts-noto-core` and `libraqm0` are needed by the synthetic Tamil/Indic OCR task. They provide the
Indic-script fonts and the shaping library that renders conjuncts correctly.

## 2. NVIDIA driver

Check that the GPU is visible on the PCI bus:

```bash
lspci | grep -i nvidia                # "VGA"/"3D controller" lines; NVSwitch "Bridge" lines on HGX
sudo ubuntu-drivers devices           # lists the driver branches available for this machine
```

Pick **one** source: the Ubuntu archive (A) or NVIDIA's repository (B). Never mix them; mixing
leaves mismatched kernel modules and libraries.

### 2a. Workstation cards: RTX 2080 Ti / 3090 / 4090

**Option A (recommended), Ubuntu archive.** These packages come with pre-built, signed kernel
modules, so they work with Secure Boot enabled:

```bash
sudo ubuntu-drivers install nvidia:595-open    # pick the highest branch >= 580 that `devices` lists
sudo reboot
```

On Ubuntu 26.04, `ubuntu-drivers autoinstall` is deprecated; use `ubuntu-drivers install`.

**Option B, NVIDIA's repository** (newest branches):

```bash
distro=$(. /etc/os-release && echo "ubuntu${VERSION_ID/./}")    # ubuntu2404 or ubuntu2604
wget https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install -y nvidia-driver-pinning-595    # optional: lock the branch
sudo apt install -y nvidia-open                  # open kernel modules (Turing and newer)
sudo reboot
```

These modules are built by DKMS. With Secure Boot on, you will be asked for a MOK password during
the install. On the next boot, choose *Enroll MOK* and enter that password.

### 2b. Data-center cards: L40S / A100 / H100 / H200

Use the **server** (data-center) driver branch. It has a longer support window, and in the Ubuntu
archive it is the branch that the Fabric Manager and NSCQ packages are built against.

**Does your machine need Fabric Manager?**

| system | Fabric Manager |
|---|---|
| PCIe cards (L40S, A100 PCIe, H100 PCIe, H100 NVL, H200 NVL), any count | not needed |
| 4-GPU HGX baseboards (direct NVLink mesh, no NVSwitch) | not needed |
| **8-GPU HGX / DGX A100, H100, H200 (NVSwitch)** | **required**: without it CUDA fails with *system not yet initialized*, even when you use a single GPU |

**Option A (recommended), Ubuntu archive:**

```bash
BR=595                                                   # highest NNN with nvidia-driver-NNN-server listed
sudo ubuntu-drivers install --gpgpu nvidia:${BR}-server
sudo apt install -y nvidia-utils-${BR}-server

# 8-GPU HGX/NVSwitch systems only:
sudo apt install -y nvidia-fabricmanager-${BR} libnvidia-nscq-${BR}
sudo apt-mark hold nvidia-driver-${BR}-server nvidia-fabricmanager-${BR} libnvidia-nscq-${BR}
sudo systemctl enable nvidia-fabricmanager
sudo reboot
```

In the Ubuntu archive, `nvidia-fabricmanager-<branch>` is built to match the
`nvidia-driver-<branch>-server` package, not the desktop `nvidia-driver-<branch>`.

**Option B, NVIDIA's repository:**

```bash
# add the cuda-keyring as in 2a, then:
sudo apt install -y nvidia-driver-pinning-595            # lock the branch FIRST
sudo apt install -y nvidia-open
sudo apt install -y cuda-drivers-fabricmanager-595       # 8-GPU HGX/NVSwitch only
sudo systemctl enable nvidia-fabricmanager
sudo reboot
```

**After the reboot (data center):**

```bash
sudo systemctl enable --now nvidia-persistenced          # keeps the driver loaded; faster CUDA start-up
systemctl status nvidia-fabricmanager                    # HGX: must be "active (running)"
nvidia-smi topo -m                                       # NVLink / NVSwitch topology
nvidia-smi -q | grep -A2 -i fabric                       # HGX: State Completed, Status Success
```

**MIG (A100 / H100 / H200).** If the GPU is partitioned, each process sees one slice. List the slices
with `nvidia-smi -L` and choose one with `CUDA_VISIBLE_DEVICES=MIG-<uuid>`. indibench sizes its
profile from the slice's memory. To use the whole GPU, disable MIG with `sudo nvidia-smi -i 0 -mig 0`
and reset or reboot.

**Shared servers.** Pin indibench to a free GPU with `CUDA_VISIBLE_DEVICES=3 indibench run ...`.

### 2c. Verify (all GPUs)

```bash
nvidia-smi                                          # driver >= 580, all GPUs listed
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
```

Optional power and noise settings for workstation cards:

```bash
sudo nvidia-smi -pm 1                  # persistence mode
sudo nvidia-smi -pl 300                # e.g. cap a 3090/4090 at 300 W (quieter, ~5-10% slower)
```

## 3. (Optional) CUDA toolkit

You only need this to *compile* CUDA code, for example flash-attn when no prebuilt wheel matches:

```bash
# with NVIDIA's repo keyring installed (2a, option B):
sudo apt install -y cuda-toolkit
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
nvcc --version
```

Ubuntu 26.04 also packages a `cuda-toolkit` (13.1) in multiverse. The toolkit's **major** version
must match the CUDA version of your PyTorch wheel (13.x) when you build extensions.

## 4. Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
cd indibench
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch torchvision                          # PyPI wheels = CUDA 13.x
uv pip install -e ".[hf,quant,api,dev,math]" fonttools
```

FlashAttention-2 works on **RTX 3090/4090, L40S, A100, H100 and H200**; skip it on a 2080 Ti:

```bash
MAX_JOBS=8 uv pip install flash-attn --no-build-isolation     # may compile for 10-60 min
```

If it fails, nothing breaks: indibench falls back to PyTorch SDPA. On Hopper, vLLM ships its own
FlashAttention-3 kernels.

### vLLM (separate environment)
vLLM pins its own PyTorch version. Keep it in its own venv so that it doesn't conflict with the hf
environment:

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python vllm --torch-backend=auto
uv pip install --python .venv-vllm/bin/python -e ".[api]"
source .venv-vllm/bin/activate     # use this env for --backend vllm
```

vLLM is the recommended backend on L40S, A100, H100 and H200, where it is often 3–10× faster than
transformers. On a **2080 Ti** it runs in fp16 with fallback kernels, and some newer architectures
fail to start on sm_75. If that happens, use `--backend hf`.

## 5. Verify the whole stack

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
indibench env -m Qwen/Qwen2.5-VL-7B-Instruct        # capabilities, auto profile, memory-fit verdict
pytest -q                                            # offline test suite (includes a tiny real HF model)
```

## 6. Hugging Face access and cache

```bash
hf auth login                         # or: export HF_TOKEN=hf_xxx
export HF_HOME=/data/hf               # put the cache on a large disk; add to ~/.bashrc
```

70B-class checkpoints take 140+ GB on disk, so on data-center machines put `HF_HOME` on local NVMe.
Several datasets are gated. Accept their terms once on huggingface.co: **MILU, FLORES+, Belebele and
IN22**. Then check that everything loads:

```bash
indibench check tamil_open            # needs no gated access
indibench check tamil -l ta           # full Tamil suite
```

## 7. First runs per GPU

The profile is detected automatically. You can force one with `--hardware-profile <name>`, or turn
profiles off with `--hardware-profile none`. A profile only fills in values you didn't set, and if
the GPU runs out of memory the hf backend halves the batch and retries.

| GPU | first run |
|---|---|
| RTX 2080 Ti | `indibench run -c configs/rtx2080ti_hf.yaml -n 50`  (3B fp16; or `-m Qwen/Qwen2.5-VL-7B-Instruct --quantization 4bit`) |
| RTX 3090 | `indibench run -c configs/rtx3090_hf.yaml -n 100`  (7B bf16 + FA2) |
| RTX 4090 | `indibench run -c configs/rtx4090_vllm.yaml -n 200`  (vLLM; 12B with `--quantization fp8`) |
| L40S | `indibench run -c configs/tamil_l40s_hf.yaml`  (or `-b vllm`; 32B with `configs/large_model_4bit.yaml`) |
| A100 40 GB | `indibench run -b vllm -m Qwen/Qwen2.5-VL-7B-Instruct -t tamil,vision_core -n 300` |
| A100 80 GB | `indibench run -c configs/a100_80g_hf.yaml` or `-c configs/indic_a100_vllm.yaml` |
| H100 | `indibench run -c configs/h100_vllm.yaml`  (32B bf16; 72B needs NVL/AWQ) |
| H200 | `indibench run -c configs/h200_vllm.yaml`  (72B FP8, 32k context, full suite) |

For vLLM runs, activate `.venv-vllm` first. On multi-GPU servers you can either run one benchmark
per GPU in parallel (`CUDA_VISIBLE_DEVICES=0 … --run-name a` and `CUDA_VISIBLE_DEVICES=1 … --run-name b`),
or shard a large model across GPUs with `--set model.tensor_parallel_size=2` (vLLM).

---

## Troubleshooting

| symptom | fix |
|---|---|
| `nvidia-smi`: *couldn't communicate with the NVIDIA driver* | Reboot after installing. If it persists with Secure Boot on, the module was not signed: complete MOK enrollment, or use the Ubuntu-archive option. |
| `Failed to initialize NVML: Driver/library version mismatch` | The driver was upgraded under a running kernel module. Reboot. |
| `torch.cuda.is_available()` is False but `nvidia-smi` works | The driver is older than 580 and too old for the CUDA 13 wheels. Upgrade the driver. Check `python -c "import torch; print(torch.version.cuda)"`. |
| HGX: `cudaErrorSystemNotReady` / *system not yet initialized* | Fabric Manager is not running or its version differs from the driver's. Check `journalctl -u nvidia-fabricmanager`, install the exact matching version, and keep both on `apt-mark hold`. |
| HGX: things broke after an unattended upgrade | The driver and Fabric Manager drifted apart. Pin the branch (`nvidia-driver-pinning-<br>` or `apt-mark hold`) and reinstall the matching pair. |
| A100/H100 shows ~10–40 GB instead of the full memory | MIG is enabled. Check `nvidia-smi -L` and disable it with `nvidia-smi -mig 0` if you want the whole GPU. |
| `no kernel image is available for execution on the device` | The GPU is older than Turing, or the torch build was compiled without your architecture. Run `indibench env`. |
| 2080 Ti gives empty, `!!!!` or NaN output | The model overflows in fp16 (common for Gemma). Use a different model or a bf16-capable GPU. |
| CUDA OOM | Lower `--batch-size`, `--max-pixels 401408`, `--max-model-len 4096` or `--max-images 2`. Use `--quantization 4bit`/`8bit`/`fp8`, or `--max-gpu-memory N` (spills to CPU RAM; slow). |
| A GPU driving the desktop has ~1 GB less free memory | Run headless (`sudo systemctl set-default multi-user.target`), use the iGPU for the display, or lower `--gpu-mem-util` for vLLM. |
| flash-attn build fails | Not needed: indibench falls back to SDPA. To build it, install the CUDA toolkit (§3) with the same major version as `torch.version.cuda`. |
| vLLM fails at start-up on a 2080 Ti | Known sm_75 limitations for some architectures. Use `--backend hf`, or try `--set model.enforce_eager=true`. |
| Indic OCR images show broken conjuncts | `libraqm0` or a Noto font is missing. Run `sudo apt install libraqm0 fonts-noto-core`. |
