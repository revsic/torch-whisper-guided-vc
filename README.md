# torch-whisper-guided-vc

Torch implementation of Whisper-guided DDPM based Voice Conversion

- DiffWave: A Versatile Diffusion Model for Audio Synthesis, Zhifeng Kong et al., 2020. [[arXiv:2009.09761](https://arxiv.org/abs/2009.09761)]
- Guided-TTS 2: A Diffusion Model for High-quality Adaptive Text-to-Speech with Untranscribed Data, Sungwon Kim et al., 2022. [[arXiv:2205.15370](https://arxiv.org/abs/2205.15370)]
- Variational Diffusion Models, Kingma et al., 2021. [[arXiv:2107.00630](https://arxiv.org/abs/2107.00630)]
- Whisper: Robust Speech Recognition via Large-Scale Weak Supervision, Radford et al., 2022. [[openai:whisper](https://cdn.openai.com/papers/whisper.pdf)]

## Requirements

Tested in python 3.7.9 conda environment.

## Usage

Download LibriTTS dataset from [openslr](https://openslr.org/60/)

To train model, run [train.py](./train.py)

```bash
python train.py \
    --data-dir /datasets/LibriTTS/train-clean-360
```

To start to train from previous checkpoint, --load-epoch is available.

```bash
python train.py \
    --data-dir /datasets/LibriTTS/train-clean-360 \
    --load-epoch 20 \
    --config ./ckpt/t1.json
```

Checkpoint will be written on TrainConfig.ckpt, tensorboard summary on TrainConfig.log.

```bash
tensorboard --logdir ./log
```

[TODO] To inference model, run [inference.py](./inference.py)

[TODO] Pretrained checkpoints are relased on [releases](https://github.com/revsic/torch-whisper-guided-vc/releases).

To use pretrained model, download files and unzip it. Followings are sample script.

```py
from wgvc import WhisperGuidedVC

ckpt = torch.load('t1_200.ckpt', map_location='cpu')
wgvc = WhisperGuidedVC.load(ckpt)
wgvc.eval()
```

## [TODO] Learning curve

## [TODO] Samples
