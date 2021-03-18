"""Utility functions for the package.

(C) 2021 Roman Werpachowski.
"""
import numpy as np


def standardise_features(features: np.ndarray) -> np.ndarray:
    """Standardises an N x D feature matrix with data points in rows.

    Subtracts a mean feature vector from every row. Divides every column by its
    *biased* standard deviation (if N > 1).

    Args:
        features: N x D feature matrix.

    Returns:
        A standardised copy.
    """
    if len(features.shape) != 2:
        raise ValueError(f"Features matrix must be 2D, got {features.shape}")
    X = features.copy()
    if not X.size:
        return X
    X -= np.mean(X, axis=0)
    if X.shape[0] > 1:
        X /= np.std(X, axis=0, ddof=0)
    return X
