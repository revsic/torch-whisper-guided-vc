from typing import List, Tuple

import numpy as np

from speechset.speeches.speechset import SpeechSet


class WavDataset(SpeechSet):
    """Waveform only dataset.
    """
    def normalize(self, _: str, speech: np.ndarray) -> np.ndarray:
        """Normalize datum.
        Args:
            _: transcription.
            speech: [np.float32; [T]], speech in range [-1, 1].
        Returns:
            speech only.
        """
        return speech

    def collate(self, bunch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [np.float32; [T]], speech signal.
        Returns:
            batch data.
                lengths: [np.long; [B]], speech lengths.
                speeches: [np.float32; [B, T]], speech signal.
        """
        # [B]
        lengths = np.array([len(s) for s in bunch])
        # [B, T]
        speeches = np.stack([
            np.pad(signal, [0, len_ - len(signal)])
            for len_, signal in zip(lengths, bunch)])
        return lengths, speeches
