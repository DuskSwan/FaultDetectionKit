from config import cfg
from data import build_dataloader, build_dataset

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

if __name__ == "__main__":
    test_dataloader()