"""
Word Matching Module

Provides MFCC-based word matching functionality for comparing audio clips
against a reference word (Temporal Feature Fingerprinting).
"""

from typing import Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine

class WordMatcher:
class WordMatcher:
    """
    Matches audio clips using MFCC (Mel-Frequency Cepstral Coefficients) similarity.
    Supports multiple reference profiles for multi-sample matching.
    """
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.references = []  # List of dicts: {name, mfcc_mean, mfcc_std, filename}

    def extract_mfcc(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20, n_fft=512, hop_length=160)
        mfcc_mean = np.mean(mfcc, axis=1)

        """
        Word Matching Module

        Provides MFCC-based word matching functionality for comparing audio clips
        against a reference word (Temporal Feature Fingerprinting).
        """

        from typing import Tuple
        import numpy as np
        import librosa
        import soundfile as sf
        from scipy.spatial.distance import cosine

        class WordMatcher:
            """
            Matches audio clips using MFCC (Mel-Frequency Cepstral Coefficients) similarity.
            Supports multiple reference profiles for multi-sample matching.
            """
            def __init__(self, sample_rate: int = 16000) -> None:
                self.sample_rate = sample_rate
                self.references = []  # List of dicts: {name, mfcc_mean, mfcc_std, filename}

            def extract_mfcc(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20, n_fft=512, hop_length=160)
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                return mfcc_mean, mfcc_std

            def add_reference(self, audio: np.ndarray, word_name: str = "target", filename: str = None) -> None:
                mfcc_mean, mfcc_std = self.extract_mfcc(audio)
                self.references.append({
                    'name': word_name,
                    'mfcc_mean': mfcc_mean,
                    'mfcc_std': mfcc_std,
                    'filename': filename
                })
                print(f"Reference word '{word_name}' added with MFCC shape: {mfcc_mean.shape}")

            def load_reference_from_file(self, filepath: str, word_name: str = "target") -> None:
                audio, sr = librosa.load(filepath, sr=self.sample_rate)
                self.add_reference(audio, word_name, filename=filepath)

            def calculate_similarity(self, audio: np.ndarray, ref) -> float:
                candidate_mfcc_mean, candidate_mfcc_std = self.extract_mfcc(audio)
                sim_mean = 1 - cosine(ref['mfcc_mean'], candidate_mfcc_mean)
                sim_std = 1 - cosine(ref['mfcc_std'], candidate_mfcc_std)
                combined_similarity = (sim_mean * 0.7 + sim_std * 0.3)
                similarity_percent = combined_similarity * 100
                scaled_similarity = (similarity_percent ** 1.5) / (100 ** 0.5)
                return scaled_similarity

            def best_match(self, audio: np.ndarray, threshold: float = 75):
                if not self.references:
                    raise ValueError("No reference profiles loaded.")
                best_name = None
                best_score = -float('inf')
                all_scores = []
                for ref in self.references:
                    score = self.calculate_similarity(audio, ref)
                    all_scores.append((ref['name'], score))
                    if score > best_score:
                        best_score = score
                        best_name = ref['name']
                return best_name, best_score, all_scores

            def clear_references(self):
                self.references = []
            reference: Reference dict with 'mean' and 'std'.
