# -*- coding: utf-8 -*-
import numpy as np
from loguru import logger

import sys
sys.path.append(".")
# from FaultDetector.measure_based import RawSignalSimilarityDetector, AEDetector
from FaultDetector.classify_based import LSTMClassifyDetector

def normal_signal(t):
    """Generate a normal signal."""
    signal = 2 * np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.random.normal(size=t.shape)
    return signal.reshape(-1, 1) # (n, 1) shape

def faulty_signal(t):
    """Generate a faulty signal."""
    signal = 3 * np.cos(2 * np.pi * 0.1 * t + 1) + 0.5 * np.random.normal(size=t.shape) + 0.5 * t
    return signal.reshape(-1, 1) # (n, 1) shape

def gen_test_signals(normal_n=10, faulty_n=7):
    """
    Generate test signals for testing. A signal is a 2D array with shape (n, m), where n is the length of the signal and m is the number of channels.
    The function generates reference signals, normal signals, and faulty signals.
    """
    t = np.linspace(0, 10, 10240)

    normal_signals = [normal_signal(t) for _ in range(normal_n)]
    normal_signals = np.stack(normal_signals, axis=0)

    faulty_signals = [faulty_signal(t) for _ in range(faulty_n)]
    faulty_signals = np.stack(faulty_signals, axis=0)
    
    logger.debug(f"normal_signals shape: {normal_signals.shape}, faulty_signals shape: {faulty_signals.shape}")
    return normal_signals, faulty_signals

def test_LSTM_detector():
    detector = LSTMClassifyDetector(
        device='cpu',
        batch_size=50,
        train_sample_n=200,
        pred_sample_n=11,
        hidden_size=32,
        num_layers=1,
        max_epochs=100,
        lr=0.0001,
        loss_name='cross_entropy',
        optimizer='adamw',
        window_size=128,
        n_classes=2,  # 0 for normal, 1 for faulty
        n_channels=1,  # Each signal has 1 channel
        hidden_sizes=[32, 16],  # Example hidden sizes for LSTM layers
    )

    # Fit the model
    num = 5
    normal_signals, faulty_signals = gen_test_signals(normal_n=10, faulty_n=10)
    train_signals = np.concatenate([normal_signals[:num], faulty_signals[:num]], axis=0)
    labels = np.concatenate([np.zeros(num), np.ones(num)], axis=0)  # 0 for normal, 1 for faulty
    detector.fit(train_signals, labels)

    # Predict on the test signals
    test_signals = np.concatenate([normal_signals[num:], faulty_signals[num:]], axis=0)
    true_labels = np.concatenate([np.zeros(len(normal_signals) - num), np.ones(len(faulty_signals) - num)], axis=0)
    pred_labels = detector.predict(test_signals)
    accuracy = np.mean(pred_labels == true_labels)
    logger.info(f"Test accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    test_LSTM_detector()