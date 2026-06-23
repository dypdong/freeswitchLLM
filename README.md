# freeswitchLLM

learn how to use freeswitch

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
