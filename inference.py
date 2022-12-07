import argparse
import os

import librosa
import torch

from wgvc import WhisperGuidedVC


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default=None)
parser.add_argument('--wav', default=None)
parser.add_argument('--sid', default=0, type=int)
parser.add_argument('--out-dir', default='./outputs')
args = parser.parse_args()

# load checkpoint
ckpt = torch.load(args.ckpt, map_location='cpu')
wgvc = WhisperGuidedVC.load(ckpt)

device = torch.device('cuda:0')
wgvc.to(device)
wgvc.eval()

# load wav
SR = wgvc.config.sr
wav, _ = librosa.load(args.wav, sr=SR)
# convert
wav = torch.tensor(wav, device=device)
sid = torch.tensor(args.sid, device=device, dtype=torch.long)

with torch.no_grad():
    # [1, T], converted
    out, _ = wgvc.forward(wav[None], sid[None])
    out = out.squeeze(dim=0).clamp(-1, 1)
    os.makedirs(args.out_dir, exist_ok=True)
    librosa.output.write_wav(
        os.path.join(args.out_dir, 'vc.wav'),
        out.cpu().numpy(),
        SR)
