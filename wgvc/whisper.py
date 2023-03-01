from typing import Optional

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
        feature_extractor = WhisperFeatureExtractor.from_pretrained(name)
        # alias
        self.fft = feature_extractor.n_fft
        self.hop = feature_extractor.hop_length
        self.samples = feature_extractor.n_samples

        self.sr = sr
        self.resample = torchaudio.transforms.Resample(sr, feature_extractor.sampling_rate)
        # [mel, fft // 2 + 1], mel filterbanks
        self.register_buffer(
            'fbank', torch.tensor(feature_extractor.mel_filters), persistent=False)
        self.register_buffer(
            'window', torch.hann_window(feature_extractor.n_fft), persistent=False)
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
            self.fft,
            self.hop,
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
            [torch.float32; [B, d_model, T' // hop]], encoded features,
                where T' = T / sr * `WhisperFeatureExtractor.sampling_rate`
                      d_model = `WhisperModel.config.d_model`
        """
        # resample
        audio = self.resample(audio)
        # T
        timesteps = audio.size(1)
        assert timesteps <= self.samples, f'audio length should be shorter than {self.samples}'
        # [B, S], zero-padding
        audio = F.pad(audio, (0, self.samples - timesteps))
        # [B, n_mels, S // hop_length]
        logmel = self.preproc(audio)
        # [B, S // hop_length, d_model], encoder only
        outputs = self.model.encoder(logmel).last_hidden_state
        # [B, d_model, T // hop_length]
        return outputs[:, :timesteps // self.hop].transpose(1, 2)

    def train(self, mode: bool = True):
        """Support only evaluation
        """
        if mode:
            import warnings
            warnings.warn('WhisperWrapper does not support training mode')
        else:
            # super call
            super().train(False)

    def state_dict(self, *args, **kwargs):
        """Do not return the state dict.
        """
        return {}

    def _load_from_state_dict(self, *args, **kwargs):
        """Do not load state dict.
        """
        pass
