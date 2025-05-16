from pathlib import Path

import numpy as np
from loguru import logger

from config import cfg
from data.nerl import load_signal_from_mat, build_dataloader
from utils import set_random_seed, initiate_cfg

import sys
sys.path.append(".")
from FaultDetector.measure_based import RawSignalSimilarityDetector, AEDetector

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


def test_raw_signal_similarity_detector():
    set_random_seed(cfg.SEED)

    ref_sample_n = cfg.DETECT.REF_SAMPLE
    test_sample_n = cfg.TEST.SAMPLE_N
    window_size = cfg.PREPROCESS.WINDOW_SIZE
    sim_method = cfg.DETECT.SIMILARITY_METHOD
    dtw_radius = cfg.DETECT.DTW_RADIUS
    outlier_method = cfg.DETECT.OUTLIER_METHOD
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


def test_AE_detector():
    set_random_seed(cfg.SEED)

    # Call the detector
    detector = AEDetector(
        device=cfg.DEVICE,
        ref_sample_n=cfg.DETECT.REF_SAMPLE,
        pred_sample_n=cfg.TEST.SAMPLE_N,
        window_size=cfg.PREPROCESS.WINDOW_SIZE,
        outlier_method=cfg.DETECT.OUTLIER_METHOD,
        signal_threshold=cfg.DETECT.SIGNAL_THRESHOLD,
        latent_dim=cfg.MODEL.AE_LATENT_DIM,
        optimizer=cfg.TRAIN.OPTIMIZER,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        max_epochs=cfg.TRAIN.MAX_EPOCHS,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        train_sample_n=cfg.TRAIN.SAMPLE_N,
        loss_name=cfg.TRAIN.LOSS_FN,
        lr=cfg.TRAIN.LR,
        hidden_dims=[16,32,64],
    )

    # Load reference signals
    normal_path = Path(r"datasets\NREL\Healthy")
    signal_paths = list(normal_path.glob("*.mat"))
    ref_signals = []
    for signal_path in signal_paths:
        signal = load_signal_from_mat(signal_path).values
        ref_signals.append(signal)
    ref_signals = np.array(ref_signals)

    detector.fit(ref_signals)

    # Load test signal
    # test_signal_path = Path(r"datasets\NREL\Damaged\D2.mat")
    test_signal_path = Path(r"datasets\NREL\Healthy\H1.mat")
    test_signal = load_signal_from_mat(test_signal_path).values
    pred_label = detector.predict(test_signal)
    true_label = "Damaged" in str(test_signal_path)
    logger.info(f"\n Pred label: {pred_label} \n True label: {true_label}.")

def detector_campare():
    """
    Compare the performance of different detectors.
    """
    set_random_seed(cfg.SEED)

    ae_detector = AEDetector(
        device=cfg.DEVICE,
        ref_sample_n=cfg.DETECT.REF_SAMPLE,
        pred_sample_n=cfg.TEST.SAMPLE_N,
        window_size=cfg.PREPROCESS.WINDOW_SIZE,
        outlier_method=cfg.DETECT.OUTLIER_METHOD,
        signal_threshold=cfg.DETECT.SIGNAL_THRESHOLD,
        latent_dim=cfg.MODEL.AE_LATENT_DIM,
        optimizer=cfg.TRAIN.OPTIMIZER,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        max_epochs=cfg.TRAIN.MAX_EPOCHS,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        train_sample_n=cfg.TRAIN.SAMPLE_N,
        loss_name=cfg.TRAIN.LOSS_FN,
        lr=cfg.TRAIN.LR,
    )

    sim_detector = RawSignalSimilarityDetector(
        ref_sample_n=cfg.DETECT.REF_SAMPLE,
        pred_sample_n=cfg.TEST.SAMPLE_N,
        window_size=cfg.PREPROCESS.WINDOW_SIZE,
        similarity_method=cfg.DETECT.SIMILARITY_METHOD,
        dtw_radius=cfg.DETECT.DTW_RADIUS,
        outlier_method=cfg.DETECT.OUTLIER_METHOD,
        signal_threshold=cfg.DETECT.SIGNAL_THRESHOLD,
    )

    # Load reference signals
    normal_dir = Path(r"datasets\NREL\Healthy")
    signal_paths = list(normal_dir.glob("*.mat"))
    ref_signals = []
    for signal_path in signal_paths:
        signal = load_signal_from_mat(signal_path).values
        ref_signals.append(signal)
    ref_signals = np.array(ref_signals)

    # Fit the detectors
    ae_detector.fit(ref_signals)
    sim_detector.fit(ref_signals)

    # Load test signals
    damaged_dir = Path(r"datasets\NREL\Damaged")
    test_signal_files = list(damaged_dir.glob("*.mat")) + list(normal_dir.glob("*.mat"))
    test_signals = []
    for signal_path in test_signal_files:
        signal = load_signal_from_mat(signal_path).values
        test_signals.append(signal)
    test_signals = np.array(test_signals) # (n_samples, window_size, n_channels)
    
    # calculate the performance of each detector
    ae_preds = ae_detector.predict(test_signals) # List[bool] of shape (n_samples, )
    sim_preds = sim_detector.predict(test_signals) 
    true_labels = ["Damaged" in str(signal_path) for signal_path in test_signal_files]
    ae_accuracy = np.mean(np.array(ae_preds) == np.array(true_labels))
    sim_accuracy = np.mean(np.array(sim_preds) == np.array(true_labels))
    logger.info(f"AE Detector Accuracy: {ae_accuracy:.2f}")
    logger.info(f"Similarity Detector Accuracy: {sim_accuracy:.2f}")


if __name__ == "__main__":
    # test_dataloader()
    # test_raw_signal_similarity_detector()
    test_AE_detector()
    # detector_campare()

