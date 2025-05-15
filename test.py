import numpy as np
from pathlib import Path
from loguru import logger

from config import cfg
from data.nerl import load_signal_from_mat, get_samples_from_signal, build_dataloader
from utils import set_random_seed, initiate_cfg

# from FaultDetector.signal_similarity import SignalSimilarityDetector
from FaultDetector.similarity_based import RawSignalSimilarityDetector

def test_dataloader():
    sample_n = cfg.TRAIN.SAMPLE_N
    window_size = cfg.PREPROCESS.WINDOW_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE

    dataloader = build_dataloader(sample_n, window_size, batch_size, is_train=True)

    for i, (X, y) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        if i == 2:
            break

def test_similarity_detector():
    set_random_seed(cfg.SEED)

    ref_sample_n = cfg.DETECT.REF_SAMPLE
    test_sample_n = cfg.TEST.SAMPLE_N
    window_size = cfg.PREPROCESS.WINDOW_SIZE
    sim_method = cfg.DETECT.SIMILARITY_METHOD
    dtw_radius = cfg.DETECT.DTW_RADIUS
    outlier_method = cfg.DETECT.OUTLIER_METHOD
    sample_abnormal_threshold = cfg.DETECT.SAMPLE_THRESHOLD
    signal_abnormal_threshold = cfg.DETECT.SIGNAL_THRESHOLD

    # Load reference signals
    normal_path = Path(r"datasets\NREL\Healthy")
    signal_paths = list(normal_path.glob("*.mat"))
    ref_signals = []
    for signal_path in signal_paths:
        signal = load_signal_from_mat(signal_path).values
        ref_signals.append(signal)
    ref_signals = np.array(ref_signals)

    # Call the detector
    detector = RawSignalSimilarityDetector(
        ref_sample_n=ref_sample_n,
        pred_sample_n=test_sample_n,
        window_size=window_size,
        similarity_method=sim_method,
        dtw_radius=dtw_radius,
        outlier_method=outlier_method,
        sample_threshold=sample_abnormal_threshold,
        signal_threshold=signal_abnormal_threshold,
    )
    detector.fit(ref_signals)

    # Load test signal
    test_signal_path = Path(r"datasets\NREL\Damaged\D2.mat")
    # test_signal_path = Path(r"datasets\NREL\Healthy\H1.mat")
    test_signal = load_signal_from_mat(test_signal_path).values
    pred_label = detector.predict(test_signal)
    true_label = "Damaged" in str(test_signal_path)
    logger.info(f"Test signal is abnormal? {pred_label}. True label: {true_label}.")


if __name__ == "__main__":
    # test_dataloader()
    test_similarity_detector()