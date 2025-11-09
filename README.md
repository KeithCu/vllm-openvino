## Installation

vLLM powered by OpenVINO supports all LLM models from the official vLLM supported models list and can perform optimal model serving on:
- All x86-64 CPUs with at least AVX2 support
- Integrated and discrete Intel® GPUs (see the OpenVINO supported GPU list)
- (Where available) Intel® NPUs via OpenVINO

> [!NOTE]
> There are no pre-built wheels or Docker images for this integration; you must install from source.

## Requirements

- Linux
- x86-64 CPU with AVX2
- Python 3.x
- For GPU/NPU acceleration:
  - Intel GPU/NPU drivers and runtime (e.g. `intel-compute-runtime`, `oneapi-level-zero`)
  - OpenVINO runtime
- Recommended / validated example:
  - Arch Linux
  - Intel Core Ultra 7 258V (Lunar Lake)
  - 32GB RAM (shared with GPU/NPU)
  - `intel_gpu_top` available for GPU monitoring

To use vLLM OpenVINO backend with a GPU, ensure your system is properly configured:
https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html

---

## Install vLLM with OpenVINO backend (Python)

There are currently no pre-built OpenVINO wheels; build and install from source.

1. Ensure Python and pip are available. For example, on Ubuntu 22.04:

```bash
sudo apt-get update -y
sudo apt-get install -y python3-pip
pip install --upgrade pip
```

2. Clone this repository:

```bash
git clone https://github.com/vllm-project/vllm-openvino.git
cd vllm-openvino
```

3. Install vLLM with OpenVINO backend:

```bash
VLLM_TARGET_DEVICE="empty" PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" python -m pip install -v .
```

> [!NOTE]
> Triton is installed by upstream vLLM on x86, but is not compatible with the OpenVINO backend. Uninstall it:
> ```bash
> python3 -m pip uninstall -y triton
> ```

---

## Prepare an OpenVINO model (required before serving)

Before starting the vLLM server with the OpenVINO backend, you MUST provide a model in OpenVINO IR or compatible exported format. Typical workflow:

1. Choose a model from Hugging Face or a local path.
2. Export/quantize it to OpenVINO format using `optimum-cli`.

### Install export tools

```bash
pip install "optimum[openvino,nncf]" transformers torch
```

### Export a Hugging Face model to OpenVINO (INT4 example)

Use `optimum-cli export openvino` to create an optimized OpenVINO model directory:

```bash
optimum-cli export openvino \
  --model <HF_MODEL_ID> \
  --task text-generation-with-past \
  --weight-format int4 \
  --sym \
  --ratio 0.8 \
  --group-size 128 \
  --trust-remote-code \
  <OUTPUT_DIR>
```

Examples:

Llama 3.2 3B Instruct:

```bash
optimum-cli export openvino \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --task text-generation-with-past \
  --weight-format int4 \
  --sym \
  --ratio 0.8 \
  --group-size 128 \
  --trust-remote-code \
  Llama3.2-3B-ov
```

Gemma 2 9B:

```bash
optimum-cli export openvino \
  --model google/gemma-2-9b-it \
  --task text-generation-with-past \
  --weight-format int4 \
  --sym \
  --ratio 0.8 \
  --group-size 128 \
  --trust-remote-code \
  Gemma2-9B-ov
```

> [!TIP]
> For maximum quality, consider `--awq --dataset wikitext2` instead of `--sym` (adds 10–30 minutes of calibration).

### Recommended models (examples)

Example, well-performing choices when exported with INT4:

| Model            | HF ID                              | Size | Quant | Quality | Speed (GPU, approx) | Use Case                    |
|------------------|------------------------------------|------|-------|---------|---------------------|-----------------------------|
| Dolphin 3.0 8B   | `dphn/Dolphin3.0-Llama3.1-8B`     | 8B   | INT4  | ★★★★★   | 25–35 t/s           | General, coding, uncensored |
| Llama 3.2 3B     | `meta-llama/Llama-3.2-3B-Instruct`| 3B   | INT4  | ★★★★☆   | 40–50 t/s           | Fast chat, edge             |
| Gemma 2 9B       | `google/gemma-2-9b-it`            | 9B   | INT4  | ★★★★★   | 20–30 t/s           | Reasoning, math             |
| Phi-3.5 Mini     | `microsoft/Phi-3.5-mini-instruct` | 3.8B | INT4  | ★★★★☆   | 35–45 t/s           | Efficient, high quality     |
| Qwen 2.5 7B      | `Qwen/Qwen2.5-7B-Instruct`        | 7B   | INT4  | ★★★★★   | 25–35 t/s           | Multilingual, strong        |

### Using the Dolphin3-ov example

This repository includes a `Dolphin3-ov/` folder as a concrete example of a prepared OpenVINO model. You can use it directly with the serving instructions below to validate your setup.

### About GGUF

- GGUF (llama.cpp) is NOT the same as OpenVINO IR and cannot be used directly.
- You can attempt conversion from a local GGUF file:

```bash
optimum-cli export openvino \
  --model ./models/llama-3.2-3b-instruct-q4_k_m.gguf \
  --task text-generation-with-past \
  --weight-format int4 \
  Llama3.2-3B-gguf-ov
```

- However, GGUF → IR may lose metadata; prefer exporting directly from the original Hugging Face model.

---

## Serve an OpenVINO model with vLLM

Once you have an exported OpenVINO model directory (e.g. `Dolphin3-ov`, `Llama3.2-3B-ov`), start the vLLM OpenAI-compatible server.

### Generic serving command

```bash
export VLLM_OPENVINO_DEVICE=GPU    # or CPU
export VLLM_OPENVINO_KVCACHE_SPACE=16
export VLLM_OPENVINO_KV_CACHE_PRECISION=i8
export VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON

python -m vllm.entrypoints.openai.api_server \
  --model <OUTPUT_DIR> \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192
```

### Quick start: Dolphin3-ov

```bash
export VLLM_OPENVINO_DEVICE=GPU
export VLLM_OPENVINO_KV_CACHE_PRECISION=i8
export VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON
export VLLM_OPENVINO_KVCACHE_SPACE=16

python -m vllm.entrypoints.openai.api_server \
  --model Dolphin3-ov \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192
```

This configuration has been validated on Arch Linux + Intel Core Ultra 7 258V (Lunar Lake).

---

## Use with OpenWebUI and OpenAI-compatible clients

### OpenWebUI

1. OpenWebUI → Admin Panel → Connections → Add OpenAI:
   - Name: `vLLM-OpenVINO`
   - API Base: `http://localhost:8000/v1`
   - API Key: leave blank or a dummy if required
2. Models → Add Custom Model:
   - Connection: `vLLM-OpenVINO`
   - Model ID: name matching the `--model` you serve (e.g. `Dolphin3-ov`, `Llama3.2-3B-ov`)

### Other OpenAI-compatible clients

Configure:
- Base URL: `http://localhost:8000/v1`
- Model: same as the `--model` directory or alias.

---

## Supported features

OpenVINO vLLM backend supports the following advanced vLLM features:

- Prefix caching (`--enable-prefix-caching`)
- Chunked prefill (`--enable-chunked-prefill`)

> [!NOTE]
> Simultaneous usage of both `--enable-prefix-caching` and `--enable-chunked-prefill` is not yet implemented.

> [!NOTE]
> `--enable-chunked-prefill` is broken on `openvino==2025.2`. To use this feature, update OpenVINO to a nightly 2025.3 build or use `openvino==2025.1`.

---

## Performance tips

### Core environment variables

- `VLLM_OPENVINO_DEVICE`
  - Target device: `CPU`, `GPU`, `GPU.X`, `NPU` (if available).
- `VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON`
  - Enable U8 weights compression during model loading when supported.
- `VLLM_USE_V1=1`
  - Enable vLLM V1 API (if desired).

### CPU performance tips

- `VLLM_OPENVINO_KVCACHE_SPACE`
  - KV cache size in GB (e.g. `40` = 40 GB). Larger values allow more parallel requests.
- `VLLM_OPENVINO_KV_CACHE_PRECISION=u8`
  - Default; reduces memory footprint.

For improved TPOT / TTFT latency, you can use chunked prefill. Example:

```bash
VLLM_OPENVINO_KVCACHE_SPACE=100 \
VLLM_OPENVINO_KV_CACHE_PRECISION=u8 \
VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
python3 vllm/benchmarks/benchmark_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  --enable-chunked-prefill \
  --max-num-batched-tokens 256
```

### GPU performance tips

- By default, GPU backend:
  - Detects available memory
  - Reserves KV cache according to `gpu_memory_utilization`
- To override:
  - `VLLM_OPENVINO_KVCACHE_SPACE=8` → 8 GB for KV cache
- Use `VLLM_OPENVINO_KV_CACHE_PRECISION` (e.g. `i8`, `fp16`) to trade off memory vs. quality.

Example best-known configuration:

```bash
VLLM_OPENVINO_DEVICE=GPU \
VLLM_OPENVINO_KV_CACHE_PRECISION=i8 \
VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
python3 vllm/benchmarks/benchmark_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json
```

---

## GPU monitoring & troubleshooting

### Confirm GPU is used

```bash
sudo intel_gpu_top
```

You should see:
- Render/3D utilization during generation
- Corresponding power usage

### If GPU/NPU not detected

Check OpenVINO devices:

```bash
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

Expected: `['CPU', 'GPU']` or `['CPU', 'GPU', 'NPU']`.

If missing (example for Arch-based systems):

```bash
sudo pacman -S intel-compute-runtime oneapi-level-zero
sudo usermod -aG render $USER
# Reboot
```

Ensure the appropriate runtime and groups are configured for your distribution.

---

## Automate with script (optional)

Example helper script to start the server:

```bash
#!/bin/bash
export VLLM_OPENVINO_DEVICE=GPU
export VLLM_OPENVINO_KV_CACHE_PRECISION=i8
export VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON
export VLLM_OPENVINO_KVCACHE_SPACE=16

python -m vllm.entrypoints.openai.api_server \
  --model "$1" \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192
```

Usage:

```bash
./start_vllm.sh Dolphin3-ov
```

---

## Limitations

- LoRA serving is not supported.
- Only decoder-style LLMs are currently supported. LLaVA and encoder-decoder models are not enabled in this integration.
- Tensor and pipeline parallelism are not currently enabled.

---

## Summary (quick reference)

- Prepare (required):
  - `optimum-cli export openvino --model <id> --task text-generation-with-past --weight-format int4 --sym --ratio 0.8 --group-size 128 <dir>`
- Serve:
  - `python -m vllm.entrypoints.openai.api_server --model <dir> --port 8000 --host 0.0.0.0`
- Integrate:
  - Point OpenWebUI / OpenAI-compatible clients to `http://localhost:8000/v1` with the chosen model ID.

Validated on Arch Linux + Intel Lunar Lake (Core Ultra 7 258V, 32GB RAM), but intended to be generic for Linux + Intel CPU/GPU setups.

**Made with ❤️ on Arch Linux + Intel Lunar Lake**  
*Tested on Core Ultra 7 258V, 32GB RAM, OpenVINO 2025.3, vLLM 0.8.4*