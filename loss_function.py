import numpy as np
import torch
import torch.nn.functional as F


class EucLoss(torch.nn.Module):

    def __init__(self, ):
        super(EucLoss, self).__init__()

    def euc_distance(self, x, y):
        distances = (x - y).pow(2).sum(1)
        return distances.sum()

    def forward(self, output1, output2):
        output1 = F.normalize(output1)
        output2 = F.normalize(output2)
        return self.euc_distance(output1, output2)


def loss_fn(x, y):
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


if __name__ == '__main__':
    x = torch.tensor(np.random.randint(0, 10, [32, 32, 15])).to("cuda:0" if torch.cuda.is_available() else "cpu")
    y = torch.tensor(np.random.randint(0, 10, [32, 32, 15])).to("cuda:0" if torch.cuda.is_available() else "cpu")
    print(x.shape)
    print(y.shape)
    loss_function = EucLoss()

    d = loss_function(x, y)
    print(len(d))

