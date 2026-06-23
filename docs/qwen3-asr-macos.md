# macOS 本地部署 Qwen3-ASR-1.7B 与 Demo 调试指南

本文档用于在本地 Mac 笔记本上拉取 GitHub 项目、创建 Python 环境、下载/加载 `Qwen/Qwen3-ASR-1.7B`，并启动命令行或 Web Demo 做语音识别调试。

> 说明：Qwen3-ASR 官方 `qwen-asr` 包优先面向 CUDA/vLLM 场景。Mac 本地调试建议先使用 `transformers` 后端；Apple Silicon 可尝试 `mps`，Intel Mac 通常使用 `cpu`，速度会明显慢于独立 GPU。

## 1. 前置条件

建议准备：

- macOS 14+（Apple Silicon M 系列优先；Intel Mac 也可用于功能验证）。
- Python 3.12（推荐用 Miniconda、Miniforge 或 `uv` 管理环境）。
- Git、Homebrew。
- 至少 16 GB 内存；1.7B 模型首次下载需要预留数 GB 磁盘空间。

安装基础工具：

```bash
# 如未安装 Homebrew，请先安装：https://brew.sh/
brew install git git-lfs ffmpeg

git lfs install
```

## 2. 从 GitHub 拉取本项目

如果你已经知道 GitHub 仓库地址，执行：

```bash
cd ~/workspace
# 将 <GITHUB_REPO_URL> 替换为本项目在 GitHub 上的真实地址，例如 git@github.com:org/freeswitchLLM.git
git clone <GITHUB_REPO_URL> freeswitchLLM
cd freeswitchLLM
```

如果你本地已经 clone 过，只需要更新当前分支：

```bash
cd ~/workspace/freeswitchLLM
git pull --ff-only
```

> 当前 Codex 运行环境中没有配置 `origin` remote，因此无法从仓库内自动读取真实 GitHub URL；请以 GitHub 页面上的 `Code` 按钮复制的 SSH/HTTPS 地址为准。

## 3. 创建本地 Python 环境

### 方式 A：使用 Conda/Miniforge（推荐）

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
python -m pip install -U pip
pip install -r requirements-qwen3-asr.txt
```

### 方式 B：使用 Python venv

```bash
python3.12 -m venv .venv-qwen3-asr
source .venv-qwen3-asr/bin/activate
python -m pip install -U pip
pip install -r requirements-qwen3-asr.txt
```

## 4. 下载模型权重（可选但推荐）

`qwen-asr` 可以在首次运行时自动下载模型。为了便于离线调试，也可以手动下载到本项目的 `models/` 目录：

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir models/Qwen3-ASR-1.7B
```

国内网络访问 Hugging Face 不稳定时，可尝试 ModelScope：

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir models/Qwen3-ASR-1.7B
```

## 5. 命令行 Demo 调试

准备一个 WAV/MP3/M4A 音频文件，例如 `samples/test.wav`，然后运行：

```bash
python scripts/qwen3_asr_macos_demo.py samples/test.wav \
  --model models/Qwen3-ASR-1.7B \
  --device auto \
  --language Chinese
```

也可以直接让程序从 Hugging Face 模型名加载：

```bash
python scripts/qwen3_asr_macos_demo.py samples/test.wav \
  --model Qwen/Qwen3-ASR-1.7B \
  --device auto
```

参数说明：

- `--device auto`：优先选择 Apple Silicon 的 `mps`，否则使用 `cpu`。
- `--language Chinese`：强制中文识别；不传则让模型自动识别语言。
- `--json-output out/asr.json`：把识别结果保存成 JSON，便于后续联调。

## 6. Web Demo 调试

`qwen-asr` 官方提供 `qwen-asr-demo` 命令。Mac 本地建议先用 transformers 后端和 CPU/MPS 调试：

```bash
qwen-asr-demo \
  --asr-checkpoint models/Qwen3-ASR-1.7B \
  --backend transformers \
  --backend-kwargs '{"device_map":"cpu","dtype":"float32","max_inference_batch_size":1,"max_new_tokens":256}' \
  --ip 127.0.0.1 \
  --port 8000
```

启动后打开：

```text
http://127.0.0.1:8000
```

> 如果 `device_map: "mps"` 在当前 PyTorch/qwen-asr 组合上不可用，请回退到 `cpu`。CPU 可用于功能验证，但长音频会较慢。

## 7. 常见问题

### 7.1 GitHub clone 很慢或失败

- 优先使用 SSH：`git@github.com:<org>/<repo>.git`。
- 确认本机已配置 SSH key：`ssh -T git@github.com`。
- 大文件需要 Git LFS：`git lfs install && git lfs pull`。

### 7.2 Python 包安装失败

先升级基础构建工具：

```bash
python -m pip install -U pip setuptools wheel
```

如果某些依赖暂不支持你的 Python 小版本，优先使用 Python 3.12 的新环境。

### 7.3 Mac 上运行慢

这是预期现象。1.7B ASR 模型在 CPU/MPS 上适合本地功能调试；生产或高并发建议使用带 NVIDIA GPU 的 Linux 服务器，并改用 vLLM 后端。

### 7.4 需要时间戳

时间戳需要额外使用 `Qwen/Qwen3-ForcedAligner-0.6B`。先下载：

```bash
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir models/Qwen3-ForcedAligner-0.6B
```

然后按官方 `qwen-asr-demo --aligner-checkpoint` 方式启用。Mac 上时间戳推理会更慢，建议只用短音频验证。
