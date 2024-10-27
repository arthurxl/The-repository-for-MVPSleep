import torch.nn.functional as F
from fusion import *

class Encoder(nn.Module):

    def __init__(self, ):
        super(Encoder, self).__init__()
        self.fs = 100
        self.input_size = 3000
        self.num_classes = 5

        # small-size CNNs
        self.conv_time = nn.Conv1d(in_channels=1,
                                   out_channels=64,
                                   kernel_size=int(self.fs / 2),
                                   stride=int(self.fs / 16),
                                   padding=0)

        self.conv_time_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, padding=0)
        self.conv_time_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, padding=0)
        self.conv_time_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, padding=0)

        self.bn_time = nn.BatchNorm1d(self.conv_time.out_channels)
        self.bn_time_1 = nn.BatchNorm1d(self.conv_time_1.out_channels)
        self.bn_time_2 = nn.BatchNorm1d(self.conv_time_2.out_channels)
        self.bn_time_3 = nn.BatchNorm1d(self.conv_time_3.out_channels)

        self.dropout_time = nn.Dropout(p=0.5)

        self.pool_time_1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.pool_time_2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # big-size CNNs
        self.conv_fre = nn.Conv1d(in_channels=1,
                                  out_channels=64,
                                  kernel_size=int(self.fs * 4),
                                  stride=int(self.fs / 2),
                                  padding=2)

        self.conv_fre_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, padding=2)
        self.conv_fre_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, padding=2)
        self.conv_fre_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, padding=2)

        self.bn_fre = nn.BatchNorm1d(self.conv_fre.out_channels)
        self.bn_fre_1 = nn.BatchNorm1d(self.conv_fre_1.out_channels)
        self.bn_fre_2 = nn.BatchNorm1d(self.conv_fre_2.out_channels)
        self.bn_fre_3 = nn.BatchNorm1d(self.conv_fre_3.out_channels)

        self.dropout_fre = nn.Dropout(p=0.5)

        self.pool_fre_1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.pool_fre_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dp = nn.Dropout(0.5)

    def forward(self, inputs):
        x1 = self.conv_time(inputs)
        x1 = F.relu(self.bn_time(x1))
        x1 = self.pool_time_1(x1)
        x1 = self.dropout_time(x1)

        x1 = self.conv_time_1(x1)
        x1 = F.relu(self.bn_time_1(x1))
        x1 = self.conv_time_2(x1)
        x1 = F.relu(self.bn_time_2(x1))
        x1 = self.conv_time_3(x1)
        x1 = F.relu(self.bn_time_3(x1))
        x1 = self.pool_time_2(x1)

        x2 = self.conv_fre(inputs)
        x2 = F.relu(self.bn_fre(x2))
        x2 = self.pool_fre_1(x2)
        x2 = self.dropout_fre(x2)

        x2 = self.conv_fre_1(x2)
        x2 = F.relu(self.bn_fre_1(x2))
        x2 = self.conv_fre_2(x2)
        x2 = F.relu(self.bn_fre_2(x2))
        x2 = self.conv_fre_3(x2)
        x2 = F.relu(self.bn_fre_3(x2))
        x2 = self.pool_fre_2(x2)

        x = torch.cat((x1, x2), dim=-1)
        x = self.dp(x)
        return x


class Autoregressor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(Autoregressor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.autoregressor = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)


    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        self.autoregressor.flatten_parameters()
        out, _ = self.autoregressor(x, h0)

        return out


class MLP_Proj(nn.Module):
    def __init__(self):
        super(MLP_Proj, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(1920, 1920),
            nn.BatchNorm1d(1920),
            nn.ReLU(),
            nn.Linear(1920, 1920)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SubModel(nn.Module):
    def __init__(self, args):
        super(SubModel, self).__init__()
        self.encoder = Encoder()
        self.autoregressor = Autoregressor()
        self.projection = MLP_Proj()
        self.args = args

    def forward(self, x):
        x = x.split(1, 1)
        output_list = []
        for data in x:
            data = data.squeeze(1)
            data = self.encoder(data).permute(0, 2, 1)
            data = self.autoregressor(data)
            data = data.contiguous().view(data.size()[0], -1)
            data = self.projection(data)
            out = data.reshape(data.size(0), -1, 128)
            output_list.append(out)
        x_output = torch.cat(output_list, 1)
        out = x_output.contiguous().view(x_output.size()[0], self.args.seq_len, -1)

        return out

class SiamModelONE(nn.Module):
    def __init__(self, args, is_train):
        super(SiamModelONE, self).__init__()
        self.online = SubModel(args)
        self.target = SubModel(args)
        self.args = args
        self.is_train = is_train
        self.prediction = nn.Sequential(
            nn.Linear(1920, 1920),
            nn.ReLU(),
            nn.Linear(1920, 1920),
            nn.ReLU(),
            nn.Linear(1920, 1920),
            nn.ReLU()
        )

        self.m = 0.1
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data.copy_(param_online.data)  # initialize
            param_target.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data = param_target.data * self.m + param_online.data * (1. - self.m)

    def trian(self, x1, x2):
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        x1_online = self.online(x1)
        x1_online = self.prediction(x1_online)

        x1_target = self.target(x1)

        x2_online = self.online(x2)
        x2_online = self.prediction(x2_online)

        x2_target = self.target(x2)
        return x1_online, x1_target, x2_online, x2_target

    def test(self, x1):
        x1_online = self.online(x1)
        return x1_online

    def forward(self, x1, x2):
        if (self.is_train):
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()
            x1_online = self.online(x1)
            x1_online = self.prediction(x1_online)

            x1_target = self.target(x1)

            x2_online = self.online(x2)
            x2_online = self.prediction(x2_online)

            x2_target = self.target(x2)
            return x1_online, x1_target, x2_online, x2_target
        else:
            x1_online = self.online(x1)
            return x1_online



class Seq_GRU(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(Seq_GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.autoregressor = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        self.autoregressor.flatten_parameters()
        out, _ = self.autoregressor(x, h0)
        return out

# class Classifier(nn.Module):
#     def __init__(self, args, length=15):
#         super(Classifier, self).__init__()
#         self.args = args
#         # self.Seq_GRU = Seq_GRU()
#         self.classifier = nn.Sequential(
#             nn.Linear(4352, 256),
#             nn.Linear(256, 64),
#             nn.Linear(64, 5),
#         )
#
#     def forward(self, x):
#         # x = self.Seq_GRU(x)
#
#         bit_fcs = []
#         x = x.reshape(self.args.batch_size, self.args.seq_len, -1)  # [batch_size,seq_len,length]
#         for i in range(self.args.seq_len):
#             xx = x[:, i, :].squeeze()  # [batch_size,length]
#             yy = self.classifier(xx)  # [batch_size,class_num]
#             yy = yy.unsqueeze(1)
#             bit_fcs.append(yy)
#         torch_bits = torch.cat(bit_fcs, 1)  # bs, seq_len, class_num
#         return torch_bits

class Classifier(nn.Module):
    def __init__(self, args, length=15):
        super(Classifier, self).__init__()
        self.args = args
        self.Seq_GRU = Seq_GRU()
        self.classifier = nn.Sequential(
            nn.Linear(4352, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 5),
        )
        self.dp = Pusion(m1=3, m2=1)

    def forward(self, x1, x2):
        x1 = self.Seq_GRU(x1)

        bit_fcs = []
        x1 = x1.reshape(self.args.batch_size, self.args.seq_len, -1)  # [batch_size,seq_len,length]
        x2 = x2.reshape(self.args.batch_size, self.args.seq_len, -1)  # [batch_size,seq_len,length]
        x = self.dp(x1, x2)
        for i in range(self.args.seq_len):
            xx = x[:, i, :].squeeze()  # [batch_size,length]
            yy = self.classifier(xx)  # [batch_size,class_num]
            yy = yy.unsqueeze(1)
            bit_fcs.append(yy)
        torch_bits = torch.cat(bit_fcs, 1)  # bs, seq_len, class_num
        return torch_bits



