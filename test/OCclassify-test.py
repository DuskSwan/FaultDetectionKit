# -*- coding: utf-8 -*-
import numpy as np
from loguru import logger

import sys
sys.path.append(".")
from FaultDetector.oneclass_based import OCSVMDetector, IsolationForestDetector


def normal_signal(t):
    """Generate a normal signal."""
    w = 2 * np.pi * 1  # 频率为 1 Hz
    # T = 1 / w
    signal = 2 * np.sin(w * t) + 0.1 * np.random.normal(size=t.shape)
    return signal.reshape(-1, 1) # (n, 1) shape

def faulty_signal(t):
    """Generate a faulty signal."""
    signal = 3 * np.cos(2 * np.pi * 0.1 * t + 1) + 0.5 * np.random.normal(size=t.shape) + 0.5 * t
    # signal = 0.5 * np.random.normal(size=t.shape) + 0.5 * t
    return signal.reshape(-1, 1) # (n, 1) shape

def gen_test_signals(normal_n=10, faulty_n=7):
    """
    Generate test signals for testing. A signal is a 2D array with shape (n, m), where n is the length of the signal and m is the number of channels.
    The function generates reference signals, normal signals, and faulty signals.
    """
    t = np.linspace(0, 10, 10240)

    normal_signals = [normal_signal(t) for _ in range(normal_n)]
    normal_signals = np.stack(normal_signals, axis=0) # (normal_n, 10240, 1) shape

    faulty_signals = [faulty_signal(t) for _ in range(faulty_n)]
    faulty_signals = np.stack(faulty_signals, axis=0) # (faulty_n, 10240, 1) shape
    
    logger.debug(f"normal_signals shape: {normal_signals.shape}, faulty_signals shape: {faulty_signals.shape}")
    return normal_signals, faulty_signals

def test_OCSVM_detector():
    detector = OCSVMDetector(
        train_sample_n=350,
        pred_sample_n=15,
        window_size=128,
        signal_threshold=0.6,  
    )

    normal_n = 10
    faulty_n = 10
    train_num = 5
    # Fit the model
    normal_signals, faulty_signals = gen_test_signals(normal_n, faulty_n)
    detector.fit(normal_signals[:train_num])

    # Predict on the test signals
    test_signals = np.concatenate([normal_signals[train_num:], faulty_signals[train_num:]], axis=0)
    true_labels = np.concatenate([np.ones(len(normal_signals) - train_num), -np.ones(len(faulty_signals) - train_num)], axis=0)
    pred_labels = detector.predict(test_signals)
    accuracy = np.mean(pred_labels == true_labels)
    logger.info(f"Test accuracy: {accuracy:.2f}")

def test_isolation_forest_detector():
    detector = IsolationForestDetector(
        train_sample_n=350,
        pred_sample_n=15,
        window_size=128,
        signal_threshold=0.6,  
    )

    normal_n = 10
    faulty_n = 10
    train_num = 5
    # Fit the model
    normal_signals, faulty_signals = gen_test_signals(normal_n, faulty_n)
    detector.fit(normal_signals[:train_num])

    # Predict on the test signals
    test_signals = np.concatenate([normal_signals[train_num:], faulty_signals[train_num:]], axis=0)
    true_labels = np.concatenate([np.ones(len(normal_signals) - train_num), -np.ones(len(faulty_signals) - train_num)], axis=0)
    pred_labels = detector.predict(test_signals)
    accuracy = np.mean(pred_labels == true_labels)
    logger.info(f"Test accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    # test_OCSVM_detector()
    test_isolation_forest_detector()