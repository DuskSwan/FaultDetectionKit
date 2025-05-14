import numpy as np
from pathlib import Path
from loguru import logger

from config import cfg
from data import build_dataloader, build_dataset
from utils import set_random_seed, initiate_cfg
from utils.similarity import calc_multi_channel_signal_similarity
from utils.outlier import is_outlier

def main():
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg, merge_file='')

    normal_sample_n = cfg.DETECT.NORMAL_SAMPLE
    test_sample_n = cfg.TEST.SAMPLE_N
    window_size = cfg.PREPROCESS.WINDOW_SIZE

    norm_signals, _ = build_dataset(normal_sample_n, window_size, is_train=True)

    unknown_signals, lables = build_dataset(test_sample_n, window_size, is_train=False)
    normal_idx = np.where(lables == 0)[0][0]
    abnormal_idx = np.where(lables == 1)[0][0]
    normal_signal = unknown_signals[normal_idx]
    abnormal_signal = unknown_signals[abnormal_idx]

    similarity_method = 'dtw'
    normal_similarity = []
    for ref_signal in norm_signals:
        similarity = calc_multi_channel_signal_similarity(ref_signal, normal_signal, method=similarity_method)
        normal_similarity.append(np.mean(similarity))
    logger.info(f"Ref similarity: {normal_similarity}")
    
    abnormal_similarity = []
    for ref_signal in norm_signals:
        similarity = calc_multi_channel_signal_similarity(ref_signal, abnormal_signal, method=similarity_method)
        abnormal_similarity.append(np.mean(similarity))

    for i,sim in enumerate(abnormal_similarity):
        is_outlier_result = is_outlier(ref_array=normal_similarity, value=sim, method='zscore')
        logger.info(f"{i}. similarity = {sim:.2f}. Is outlier? {is_outlier_result}")
    

if __name__ == "__main__":
    main()