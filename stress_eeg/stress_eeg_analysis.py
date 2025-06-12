"""Stress Level Analysis from EEG signals.
This script demonstrates how you might load EEG data from .mat files, extract simple features
(power in canonical frequency bands) and train a machine learning model to classify stress vs
relaxation states.

The dataset path and structure will depend on the SAM-40 dataset. Adjust variable names as
needed. This is a template for further development.
"""

import os
from typing import Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class StressEEGAnalyzer:
    """Pipeline to train a stress classifier from EEG .mat files."""

    def __init__(self, data_dir: str, sfreq: int = 256) -> None:
        self.data_dir = data_dir
        self.sfreq = sfreq
        self.model = RandomForestClassifier(random_state=42)

    def _load_mat_file(self, path: str) -> Tuple[np.ndarray, int]:
        """Load one subject's data from a .mat file.

        The SAM-40 dataset stores raw EEG data and labels inside the MAT file.
        Adjust the keys (e.g., 'EEG', 'label') to match the dataset structure.
        """
        mat = loadmat(path)
        data = mat.get("EEG")  # shape (n_channels, n_samples)
        label = mat.get("label", np.array([0]))[0]
        return data, int(label)

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all .mat files and stack them into arrays."""
        features = []
        labels = []
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".mat"):
                data, label = self._load_mat_file(os.path.join(self.data_dir, fname))
                feat = self.extract_features(data)
                features.append(feat)
                labels.append(label)
        return np.array(features), np.array(labels)

    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Compute band power features using Welch's method."""
        freqs, psd = welch(data, fs=self.sfreq, nperseg=self.sfreq*2)
        band_limits = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }
        band_power = []
        for low, high in band_limits.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            power = psd[:, idx].mean(axis=1)
            band_power.append(power)
        # flatten features (channels x bands)
        return np.concatenate(band_power, axis=0)

    def train(self) -> None:
        X, y = self.load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))

    def predict(self, data: np.ndarray) -> int:
        feat = self.extract_features(data)
        return int(self.model.predict(feat.reshape(1, -1))[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stress detection from EEG data")
    parser.add_argument("data_dir", help="Directory containing .mat files")
    args = parser.parse_args()

    analyzer = StressEEGAnalyzer(args.data_dir)
    analyzer.train()
