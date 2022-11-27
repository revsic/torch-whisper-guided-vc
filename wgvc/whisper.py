from typing import Dict, Optional

import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor


class WhisperWrapper(nn.Module):
    """Wrapping huggingface open-ai, Whisper
    """
    DEFAULT = 'openai/whisper-base'

    OUT_CHANNELS: int = ...

    def __init__(self,
                 name: Optional[str] = None,
                 sr: int = 16000):
        """Load the Whisper pretrained model.
        Args:
            name: name of the model, default use facebook XLSR-53.
            sr: sample rates of the input audio, default 16khz for whisper-base.
        """
        super().__init__()
        name = name or WhisperModel.DEFAULT
        self.model = WhisperModel.from_pretrained(name)
        self.preproc = WhisperFeatureExtractor.from_pretrained(name)
        
        self.sr = sr
        self.resample = torchaudio.transforms.Resample(sr, self.preproc.sampling_rate)
        # mel filterbanks
        self.register_buffer(
            'fbank', torch.tensor(self.preproc.mel_filters), persistent=False)
        self.register_buffer(
            'window', torch.hann_window(self.preproc.n_fft), persistent=False)
        self.eval()

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract the features from audio.
        Args:
            audio: [torch.float32; [B, T]], audio, [-1, 1]-ranged.
        Returns:
            ...
        """
        # B
        bsize = audio.shape[0]
        # [B, T]
        audio = self.resample(audio)
        # [B, n_fft // 2 + 1, T'(=T // hop_length)]
        stft = torch.stft(
            audio,
            self.preproc.n_fft,
            self.preproc.hop_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            return_complex=True)
        # [B, n_mels, T']
        mel = self.fbank @ (stft.abs() ** 2)
        # [B, n_mels, T']
        logmel = torch.log10(mel.clamp_min(1e-10))
        # normalization
        logmel = torch.maximum(logmel, logmel.max() - 8.)
        logmel = (logmel + 4.) / 4.
        # generation
        start = torch.fill(
            self.model.config.decoder_start_token_id, (bsize, 1))
        return self.model(logmel, decoder_input_ids=start)
    
    def train(self, mode: bool = True):
        """Support only evaluation
        """
        if mode:
            import warnings
            warnings.warn('WhisperWrapper does not support training mode')
        else:
            # super call
            super().train(False)

    def load_state_dict(self,
                        state_dict: Dict[str, torch.Tensor],
                        strict: bool = True):
        """Do not load state dict.
        """
        pass
