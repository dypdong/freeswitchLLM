# freeswitchLLM

learn how to use freeswitch

## Qwen3-ASR-1.7B 本地部署

本仓库提供了在本地 Mac 笔记本上拉取 GitHub 项目、部署 `Qwen/Qwen3-ASR-1.7B`、运行命令行 Demo 和 Web Demo 的操作指南：

- [macOS 本地部署 Qwen3-ASR-1.7B 与 Demo 调试指南](docs/qwen3-asr-macos.md)
- 命令行示例脚本：[`scripts/qwen3_asr_macos_demo.py`](scripts/qwen3_asr_macos_demo.py)
- 依赖文件：[`requirements-qwen3-asr.txt`](requirements-qwen3-asr.txt)
=======
## Qwen3-ASR-1.7B local demo

This repository includes a local Qwen3-ASR-1.7B deployment/debugging guide and a CLI transcription demo:

- Guide: [`docs/qwen3-asr-local-demo.md`](docs/qwen3-asr-local-demo.md)
- Demo script: [`scripts/qwen3_asr_demo.py`](scripts/qwen3_asr_demo.py)
- Python dependencies: [`requirements-qwen3-asr.txt`](requirements-qwen3-asr.txt)

Quick start:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-qwen3-asr.txt
python scripts/qwen3_asr_demo.py ./samples/example.wav --language Chinese
```
