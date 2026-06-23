# Qwen3-ASR-1.7B 本地部署与 Demo 调试指南

本项目提供一个最小可运行的 Qwen3-ASR-1.7B 本地调试入口：

- `requirements-qwen3-asr.txt`：安装官方 `qwen-asr` Python 包及基础依赖。
- `scripts/qwen3_asr_demo.py`：命令行转写 Demo，支持本地音频/URL、语言强制指定、JSON 输出和可选时间戳。

> 参考官方模型卡：Qwen3-ASR-1.7B 支持离线/流式 ASR，覆盖 30 种语言和 22 种中文方言；官方建议使用隔离的 Python 3.12 环境，并通过 `qwen-asr` 包或 vLLM 部署。

## 1. 环境准备

建议使用带 CUDA 的 Linux 主机。1.7B 模型建议使用 NVIDIA GPU；如显存不足，可降低 `--batch-size` 或改用 0.6B 模型做联调。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-qwen3-asr.txt
```

如果需要 vLLM 或流式能力，请额外安装官方扩展：

```bash
pip install -U 'qwen-asr[vllm]'
```

## 2. 模型下载

运行时可以直接使用模型 ID 自动下载：

```bash
python scripts/qwen3_asr_demo.py samples/example.wav --language Chinese
```

在无法运行时下载权重的环境，先手动下载到本地目录：

```bash
# Hugging Face
pip install -U 'huggingface_hub[cli]'
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./models/Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./models/Qwen3-ForcedAligner-0.6B

# 中国大陆环境可改用 ModelScope
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./models/Qwen3-ForcedAligner-0.6B
```

## 3. 命令行 Demo

自动识别语言：

```bash
python scripts/qwen3_asr_demo.py ./samples/example.wav
```

指定中文并保存 JSON：

```bash
python scripts/qwen3_asr_demo.py ./samples/example.wav \
  --model ./models/Qwen3-ASR-1.7B \
  --language Chinese \
  --output ./outputs/example.zh.json
```

开启时间戳：

```bash
python scripts/qwen3_asr_demo.py ./samples/example.wav \
  --model ./models/Qwen3-ASR-1.7B \
  --timestamps \
  --aligner-model ./models/Qwen3-ForcedAligner-0.6B \
  --output ./outputs/example.timestamps.json
```

显存紧张时可调整：

```bash
python scripts/qwen3_asr_demo.py ./samples/example.wav \
  --batch-size 1 \
  --max-new-tokens 128
```

## 4. Gradio Web Demo

安装 `qwen-asr` 后可直接启动官方 Web UI：

```bash
qwen-asr-demo \
  --asr-checkpoint ./models/Qwen3-ASR-1.7B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"device_map":"cuda:0","dtype":"bfloat16","max_inference_batch_size":8,"max_new_tokens":256}' \
  --ip 0.0.0.0 --port 8000
```

浏览器打开 `http://<服务器IP>:8000`。如果需要远程麦克风录音，浏览器通常要求 HTTPS，可按官方参数添加 `--ssl-certfile` 和 `--ssl-keyfile`。

## 5. vLLM 服务化

适合需要 API 调试或更高吞吐的场景：

```bash
qwen-asr-serve ./models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 \
  --port 8000
```

请求示例：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{
      "role": "user",
      "content": [{"type":"audio_url","audio_url":{"url":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}]
    }]
  }'
```

## 6. 常见问题

- `CUDA out of memory`：降低 `--batch-size`、`--max-new-tokens`，或使用 Qwen3-ASR-0.6B 做功能联调。
- 权重下载失败：先用 Hugging Face CLI 或 ModelScope 下载到 `./models/`，再通过 `--model` 指向本地目录。
- 麦克风不可用：远程浏览器通常需要 HTTPS；本地命令行 Demo 可先用音频文件排查模型链路。
- 长音频输出截断：增大 `--max-new-tokens`，必要时切分音频后批量处理。
