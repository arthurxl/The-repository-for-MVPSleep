import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float,default=0.0001)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--seq-len', type=int, default=3)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--device', type=int, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--unit_sec', type=float, default=0.01)
    parser.add_argument('--sample_rate', type=int, default=100)

    args_parsed = parser.parse_args()
    return args_parsed