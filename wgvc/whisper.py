from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor


class WhisperWrapper(nn.Module):
    """Wrapping huggingface open-ai, Whisper
    """
    DEFAULT = 'openai/whisper-base'

    def __init__(self,
                 name: Optional[str] = None,
                 sr: int = 16000):
        """Load the Whisper pretrained model.
        Args:
            name: name of the model, default use facebook XLSR-53.
            sr: sample rates of the input audio, default 16khz for whisper-base.
        """
        super().__init__()
        name = name or WhisperWrapper.DEFAULT
        self.model = WhisperModel.from_pretrained(name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(name)

        self.sr = sr
        self.resample = torchaudio.transforms.Resample(sr, self.feature_extractor.sampling_rate)
        # [n_mels, n_fft // 2 + 1], mel filterbanks
        self.register_buffer(
            'fbank', torch.tensor(self.feature_extractor.mel_filters), persistent=False)
        self.register_buffer(
            'window', torch.hann_window(self.feature_extractor.n_fft), persistent=False)
        self.eval()

    def preproc(self, audio: torch.Tensor) -> torch.Tensor:
        """Preprocess the input audios, convert to mel-spectrogram.
        Args:
            audio: [torch.float32; [B, T]], audio, [-1, 1]-ranged.
        Returns:
            [torch.float32; [B, n_mels, T // hop_length]], log-mel spectrogram.
        """
        # [B, n_fft // 2 + 1, T'(=T // hop_length)]
        stft = torch.stft(
            audio,
            self.feature_extractor.n_fft,
            self.feature_extractor.hop_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            return_complex=True)
        # [B, n_mels, T'], drop last (ref:openai/whisper.audio)
        mel = self.fbank @ (stft[..., :-1].abs() ** 2)
        # [B, n_mels, T']
        logmel = mel.clamp_min(1e-10).log10()
        # normalization
        logmel = torch.maximum(
            logmel, logmel.amax(dim=(-1, -2), keepdim=True) - 8.)
        return (logmel + 4.) / 4.

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract the features from audio.
        Args:
            audio: [torch.float32; [B, T]], audio, [-1, 1]-ranged.
        Returns:
            [torch.float32; [B, d_model, T // hop_length]], encoded features,
                where d_model = `self.model.config.d_model`
                      hop_length = `self.feature_extractor.hop_length`
        """
        # alias
        n_samples = self.feature_extractor.n_samples
        hop_length = self.feature_extractor.hop_length
        # resample
        audio = self.resample(audio)
        # T
        timesteps = audio.size(1)
        assert timesteps <= n_samples, f'audio length should be shorter than {n_samples}'
        # [B, S], zero-padding
        audio = F.pad(audio, (0, n_samples - timesteps))
        # [B, n_mels, S // hop_length]
        logmel = self.preproc(audio)
        # [B, S // hop_length, d_model], encoder only
        outputs = self.model.encoder(logmel).last_hidden_state
        # [B, d_model, T // hop_length]
        return outputs[:, :timesteps // hop_length].transpose(1, 2)

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
