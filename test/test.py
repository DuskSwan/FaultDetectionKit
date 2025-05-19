# -*- coding: utf-8 -*-
import numpy as np
from loguru import logger

import sys
sys.path.append(".")
from FaultDetector.measure_based import RawSignalSimilarityDetector, AEDetector

def normal_signal(t):
    """Generate a normal signal."""
    signal = 2 * np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.random.normal(size=t.shape)
    return signal.reshape(-1, 1)

def faulty_signal(t):
    """Generate a faulty signal."""
    signal = 2 * np.cos(2 * np.pi * 0.1 * t + 1) + 1 * np.random.normal(size=t.shape) + 0.05 * t
    return signal.reshape(-1, 1)
    
def gen_test_signals(ref_n=5, normal_n=1, faulty_n=1):
    """
    Generate test signals for testing. A signal is a 2D array with shape (n, m), where n is the length of the signal and m is the number of channels.
    The function generates reference signals, normal signals, and faulty signals.
    """
    t = np.linspace(0, 10, 1000)

    ref_signals = [normal_signal(t) for _ in range(ref_n)]
    ref_signals = np.stack(ref_signals, axis=0)

    normal_signals = [normal_signal(t) for _ in range(normal_n)]
    normal_signals = np.stack(normal_signals, axis=0)

    faulty_signals = [faulty_signal(t) for _ in range(faulty_n)]
    faulty_signals = np.stack(faulty_signals, axis=0)
    
    logger.debug(f"ref_signals shape: {ref_signals.shape}, normal_signals shape: {normal_signals.shape}, faulty_signals shape: {faulty_signals.shape}")
    return ref_signals, normal_signals, faulty_signals


def test_raw_signal_similarity_detector():

    ref_sample_n = 10
    test_sample_n = 5
    window_size = 32
    sim_method = "dtw"
    dtw_radius = None
    outlier_method = "zscore"
    signal_abnormal_threshold = 0.5

    ref_signals, normal_signals, faulty_signals = gen_test_signals(ref_n=5, normal_n=1, faulty_n=1)

    # Call the detector
    detector = RawSignalSimilarityDetector(
        ref_sample_n=ref_sample_n,
        pred_sample_n=test_sample_n,
        window_size=window_size,
        similarity_method=sim_method,
        dtw_radius=dtw_radius,
        outlier_method=outlier_method,
        signal_threshold=signal_abnormal_threshold,
    )
    detector.fit(ref_signals)

    # Load test signal
    test_signals = np.concatenate((normal_signals, faulty_signals), axis=0)
    pred_label = detector.predict(test_signals)
    true_label = [False, True]
    logger.info(f"Test signal is abnormal? {pred_label}. True label: {true_label}.")


def test_AE_detector():

    # Call the detector
    detector = AEDetector(
        device="cuda",
        ref_sample_n=10,
        pred_sample_n=3,
        window_size=32,
        outlier_method="zscore",
        signal_threshold=0.5,
        latent_dim=16,
        optimizer="adamw",
        batch_size=32,
        max_epochs=10,
        num_workers=0,
        train_sample_n=100,
        loss_name="mse",
        lr=1e-3,
        hidden_dims=[16,32,64],
    )

    ref_signals, normal_signals, faulty_signals = gen_test_signals(ref_n=5, normal_n=1, faulty_n=1)

    detector.fit(ref_signals)

    test_signals = np.concatenate((normal_signals, faulty_signals), axis=0)
    pred_label = detector.predict(test_signals)
    true_label = [False, True]
    logger.info(f"Test signal is abnormal? {pred_label}. True label: {true_label}.")


if __name__ == "__main__":
    test_raw_signal_similarity_detector()
    test_AE_detector()

