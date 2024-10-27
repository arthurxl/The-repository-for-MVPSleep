import torchaudio.transforms as AT
import numpy as np
import torch
from torch.utils.data import Dataset


class NoseqDataset(Dataset):

    def __init__(self, args, data_path, tfms_ONE, tfms_TWO, use_librosa=False):
        self.data_path = data_path
        self.data, self.label, self.sr = self.load_all_npz_files(self.data_path)
        self.tfms_ONE = tfms_ONE
        self.tfms_TWO = tfms_TWO
        self.unit_length = int(args.unit_sec * args.sample_rate)
        self.to_melspecgram = AT.MelSpectrogram(
            sample_rate=100,
            n_fft=256,
            win_length=256,
            hop_length=104,
            n_mels=129,
            # f_min=cfg.f_min,
            # f_max=cfg.f_max,
            power=2,
        )

    def load_one_npz_file(self, data_path):
        with np.load(data_path, allow_pickle=True) as f:
            data = f["x"]  #[:, :, 1]
            # print(data.shape)
            labels = f["y"]
            sampling_rate = f["fs"]

        return data, labels, sampling_rate

    def load_all_npz_files(self, data_path_lists):
        all_data = []
        all_labels = []
        all_sr = []
        for tmp_path in data_path_lists:
            print('Loading {}...'.format(tmp_path))
            tmp_data, tmp_label, tmp_sr = self.load_one_npz_file(tmp_path)

            tmp_data = tmp_data.astype(np.float32)
            all_data.append(tmp_data)
            tmp_label = tmp_label.astype(np.int64)
            all_labels.append(tmp_label)
            tmp_sr = tmp_sr.astype(np.int64)
            all_sr.append(tmp_sr)
        all_data = np.vstack(all_data).reshape(-1, 1, 3000)

        all_labels = np.hstack(all_labels)
        all_sr = np.hstack(all_sr)
        return all_data, all_labels, all_sr


    def __getitem__(self, idx):
        data = self.data[idx]

        label = self.label[idx]
        data = torch.Tensor(data)
        lms = (self.to_melspecgram(data) + torch.finfo().eps).log().unsqueeze(0)

        if self.tfms_TWO:
            lms = self.tfms_TWO(lms)

        lms1, lms2 = lms[0], lms[1]

        if label is not None:

            return data, lms1, lms2, label

        return data, lms1, lms2

    def __len__(self):
        return len(self.label)


class SeqDataset(Dataset):
    def __init__(self, data_path, seq_len=20, tfms_ONE=None, tfms_TWO=None):
        self.data_path = data_path
        # self.tfms_ONE = tfms_ONE
        self.tfms_TWO = tfms_TWO

        data, labels = self.load_npz_list_files(self.data_path)
        self.x_list = []
        self.y_list = []
        for i in range(len(data)):
            data_len = data[i].shape[0]
            num_elems = (data_len // seq_len) * seq_len
            self.x_list.append(data[i][:num_elems])
            self.y_list.append(labels[i][:num_elems])
        self.x_list = [np.split(x, x.shape[0] // seq_len)
                       for x in self.x_list]
        self.y_list = [np.split(y, y.shape[0] // seq_len)
                       for y in self.y_list]
        self.x_list = [item for sublist in self.x_list for item in sublist]
        self.y_list = [item for sublist in self.y_list for item in sublist]
        self.to_melspecgram = AT.MelSpectrogram(
            sample_rate=100,
            n_fft=256,
            win_length=256,
            hop_length=104,
            n_mels=129,
            # f_min=cfg.f_min,
            # f_max=cfg.f_max,
            power=2,
        )

    def load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        data = np.squeeze(data)
        data = data[:, np.newaxis, :]
        return data, labels, sampling_rate

    def load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self.load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)
            data.append(tmp_data)
            labels.append(tmp_labels)
        return data, labels

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):

        xxx = torch.Tensor(self.x_list[idx])
        output_list1 = []
        output_list2 = []
        for data in xxx:
            # data = data.squeeze(1)
            lms = (self.to_melspecgram(data) + torch.finfo().eps).log().unsqueeze(0)
            if self.tfms_TWO:
                # data_ONE = self.tfms_ONE(data)
                lms = self.tfms_TWO(lms)
            lms1, lms2 = lms[0], lms[1]
            output_list1.append(lms1)
            output_list2.append(lms2)
        output1 = torch.cat(output_list1, 0)  # [batch_size,length*seq_len,channel]
        output2 = torch.cat(output_list2, 0)  # [batch_size,length*seq_len,channel]
        return xxx, output1, output2, torch.LongTensor(self.y_list[idx])


