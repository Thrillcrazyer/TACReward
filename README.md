# TACReward
About Code release for "Reasoning-Aware Proxy Reward Model using Process Mining"


## Getting Started

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Training

```bash
scripts/train.sh
```

### Evaluation

```bash
scripts/train.sh
```


## Acknoewledgments

- [trl](https://github.com/huggingface/trl?tab=Apache-2.0-1-ov-file)
- [Deepmath103k](https://github.com/zwhe99/DeepMath/)
- [PM4Py](https://github.com/process-intelligence-solutions/pm4py)