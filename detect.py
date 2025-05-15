import numpy as np
from pathlib import Path
from loguru import logger

from config import cfg
from data import build_dataloader, build_dataset, get_samples_from_signal
from utils import set_random_seed, initiate_cfg
from utils.similarity import calc_multi_channel_signal_similarity
from utils.outlier import is_outlier

def calc_ref_similarity(ref_samples: np.ndarray, method: str, **kwargs) -> np.ndarray:
    '''计算参考信号片段之间的相似度分布'''
    n = len(ref_samples)
    ref_similarities = []

    ij_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    calc_pairs = np.random.choice(len(ij_pairs), size=min(2*n, len(ij_pairs)), replace=False)
    for idx in calc_pairs:
        i, j = ij_pairs[idx]
        similarity = calc_multi_channel_signal_similarity(ref_samples[i], ref_samples[j], method=method, **kwargs)
        ref_similarities.append(similarity)
        
    return np.array(ref_similarities)

def check_sample(ref_samples: np.ndarray, 
                 sample: np.array, 
                 sim_method: str,
                 normal_similarites: np.ndarray, 
                 outlier_method:str,
                 abnormal_threshold: float,
                 **kwargs
                ) -> bool:
    '''
    检查样本相对于参考样本是否为离群值
    '''
    similarites = []
    for ref_signal in ref_samples:
        similarity = calc_multi_channel_signal_similarity(ref_signal, sample, method=sim_method, **kwargs)
        similarites.append(similarity)
    outliter_cnt = 0
    for i, sim in enumerate(similarites):
        is_outlier_result = is_outlier(ref_array=normal_similarites, value=sim, method=outlier_method)
        # logger.info(f"{i}. similarity = {sim:.2f}. Is outlier? {is_outlier_result}")
        outliter_cnt += 1 if is_outlier_result else 0
    outliter_rate = outliter_cnt / len(similarites)
    logger.info(f"Outlier rate: {outliter_rate:.2f}")

    if outliter_rate > abnormal_threshold:
        logger.info(f"Sample is abnormal. Outlier rate: {outliter_rate:.2f}")
        return True
    else:
        logger.info(f"Sample is normal. Outlier rate: {outliter_rate:.2f}")
        return False

def check_signal(signal: np.ndarray, 
                 window_size: int,
                 ref_signals: np.ndarray, 
                 sim_method: str,
                 normal_similarites: np.ndarray, 
                 outlier_method:str,
                 sample_abnormal_threshold: float,
                 signal_abnormal_threshold: float,
                 **kwargs
                ) -> bool:
    '''
    检查信号是否为离群值
    '''
    samples = get_samples_from_signal(signal, window_size=window_size)
    abnormal_cnt = 0
    for sample in samples:
        is_abnormal = check_sample(ref_signals, sample, sim_method, normal_similarites, outlier_method, sample_abnormal_threshold, **kwargs)
        abnormal_cnt += 1 if is_abnormal else 0
    abnormal_rate = abnormal_cnt / len(samples)
    logger.info(f"Abnormal rate: {abnormal_rate:.2f}")
    if abnormal_rate > signal_abnormal_threshold:
        logger.info(f"Signal is abnormal. Abnormal rate: {abnormal_rate:.2f}")
        return True
    else:
        logger.info(f"Signal is normal. Abnormal rate: {abnormal_rate:.2f}")
        return False

def check():
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg, merge_file='')

    ref_sample_n = cfg.DETECT.REF_SAMPLE
    test_sample_n = cfg.TEST.SAMPLE_N
    window_size = cfg.PREPROCESS.WINDOW_SIZE
    sim_method = cfg.DETECT.SIMILARITY_METHOD
    dtw_radius = cfg.DETECT.DTW_RADIUS
    outlier_method = cfg.DETECT.OUTLIER_METHOD
    sample_abnormal_threshold = cfg.DETECT.SAMPLE_THRESHOLD
    signal_abnormal_threshold = cfg.DETECT.SIGNAL_THRESHOLD

    logger.info("Loading reference samples...")
    norm_samples, _ = build_dataset(ref_sample_n, window_size, is_train=True)
    norm_similarities = calc_ref_similarity(norm_samples, method=sim_method, dtw_radius=dtw_radius)
    logger.info(f"Normal samples similarity refence value: {np.mean(norm_similarities):.2f}")

    logger.info("Loading test samples...")
    unknown_samples, labels = build_dataset(test_sample_n, window_size, is_train=False, is_random=True)
    
    pred_right_cnt = 0
    for i, sample in enumerate(unknown_samples):
        logger.info(f"Checking sample {i}...")
        is_abnormal = check_sample(norm_samples, sample, sim_method, norm_similarities, outlier_method, sample_abnormal_threshold, dtw_radius=dtw_radius)
        true_is_abnormal = True if labels[i] == 1 else False
        logger.info(f"Sample {i}: predict right? {is_abnormal==true_is_abnormal}.")
        pred_right_cnt += 1 if is_abnormal == true_is_abnormal else 0
    pred_right_rate = pred_right_cnt / len(unknown_samples)
    logger.info(f"Pred right rate: {pred_right_rate:.2f}")
    

if __name__ == "__main__":
    check()