import time
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from config import parse_args
from dataset import SeqDataset
from modelONE import SiamModelONE, Classifier
from modelTWO import SiamModelTWO
from train import train_feature, train_class_epoch, test_class_epoch
from loss_function import EucLoss
from data_augmentation import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    args.time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    args.out_dir = 'exCH1/' + args.time

    all_data_path = glob.glob()  #路径

    n_folds = 10
    kf = KFold(n_splits=n_folds)
    results_fold = []

    cm_sum = torch.zeros(5, 5)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
        f.write("batch_size:" + str(args.batch_size) + "\n")
        f.write("learning_rate:" + str(args.learning_rate) + "\n")
        f.write("prediction_step:" + str(args.prediction_step) + "\n")
        f.write("seq_len:" + str(args.seq_len) + "\n")

    for i in range(0, n_folds):
        print('第{}次训练'.format(i + 1))
        args.current_epoch = args.start_epoch
        args.folds = str(i + 1)

        model_ONE_feature = SiamModelONE(args, is_train=True).cuda()
        model_ONE_feature.apply(weights_init)
        ONE_params = model_ONE_feature.parameters()
        model_TWO_feature = SiamModelTWO(args, is_train=True).cuda()
        model_TWO_feature.apply(weights_init)
        TWO_params = model_TWO_feature.parameters()
        optimizer_ONE_feature = torch.optim.Adam(ONE_params, lr=args.learning_rate)
        optimizer_TWO_feature = torch.optim.Adam(TWO_params, lr=args.learning_rate)

        model_class = Classifier(args)
        model_class = model_class.to(args.device)
        model_class.apply(weights_init)
        params = model_class.parameters()
        optimizer_class = torch.optim.Adam(params, lr=args.learning_rate)


        lossFunc_feature = EucLoss()

        lossFunc_class = torch.nn.CrossEntropyLoss()

        for i_fold, (train_index, test_index) in enumerate(kf.split(all_data_path)):
            if i_fold == i:
                all_test_npz_path = []
                for k in range(0, len(test_index)):
                    all_test_npz_path.append(all_data_path[test_index[k]])
                all_train_npz_path = np.setdiff1d(all_data_path, all_test_npz_path)


        tfms_TWO = AugmentationTWO(2 * len(all_train_npz_path))
        train_feature_dataset = SeqDataset(all_train_npz_path, args.seq_len, tfms_TWO=tfms_TWO)
        train_dataset = SeqDataset(all_train_npz_path, args.seq_len, tfms_TWO=tfms_TWO)
        test_dataset = SeqDataset(all_test_npz_path, args.seq_len, tfms_TWO=tfms_TWO)

        kwargs = {'pin_memory': True}
        train_feature_loader = torch.utils.data.DataLoader(train_feature_dataset, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True, **kwargs)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  drop_last=True, **kwargs)

        best_acc = 0
        args.current_epoch = 0
        save_train_acc = []
        save_test_acc = []
        save_train_loss = []
        save_test_loss = []
        for idx in range(0, args.num_epochs):
            print("-----------------------epoch:{}-----------------------".format(idx + 1))

            loss_feature = train_feature(args, model_ONE_feature, model_TWO_feature, optimizer_ONE_feature,
                                         optimizer_TWO_feature, train_feature_loader, lossFunc_feature)
            context_ONE_model = model_ONE_feature.eval()
            context_TWO_model = model_TWO_feature.eval()

            acc_train, loss_class_train = train_class_epoch(args, train_loader, context_ONE_model, context_TWO_model,
                                                            model_class,
                                                            optimizer_class,
                                                            lossFunc_class)
            acc_test, loss_class_test, mf1, per_class_f1, k, cm, per_class_acc = test_class_epoch(args, test_loader,
                                                                                                  context_ONE_model,
                                                                                                  context_TWO_model,
                                                                                                  model_class,
                                                                                                  lossFunc_class)
            args.current_epoch = idx + 1
            args.mf1 = mf1
            args.per_class_f1 = per_class_f1
            args.k = k
            args.cm = cm
            args.per_class_acc = per_class_acc
            outout = os.path.join(args.out_dir, args.folds)
            if not os.path.exists(outout):
                os.makedirs(outout)

            save_train_acc.append(acc_train)
            save_test_acc.append(acc_test)
            save_train_loss.append(loss_class_train)
            save_test_loss.append(loss_class_test)
            print("acc_train:{} ,loss_train:{}".format(acc_train, loss_class_train))
            print("acc_test:{} ,loss_test:{}".format(acc_test, loss_class_test))
            if (acc_test > best_acc):
                best_acc = acc_test
                args.best_acc = best_acc
                print("best_acc!!!:{}".format(best_acc))

        print("Final_best_acc!!!:{}".format(best_acc))
        plt_x1 = range(1, args.num_epochs + 1)
        plt_x2 = range(1, args.num_epochs + 1)
        save_acc_1 = []
        save_acc_2 = []
        for j in range(0, len(save_train_acc)):
            save_acc_1.append(save_train_acc[j])
        for j in range(0, len(save_test_acc)):
            save_acc_2.append(save_test_acc[j])
        plt.plot(plt_x1, save_acc_1, 'b')
        plt.plot(plt_x1, save_acc_2, 'r')
        out1 = os.path.join(args.out_dir, args.folds, 'acc.png')
        plt.savefig(out1)
        plt.pause(1)
        plt.close()

        save_loss_1 = []
        save_loss_2 = []
        for j in range(0, len(save_train_loss)):
            save_loss_1.append(save_train_loss[j])
        for j in range(0, len(save_test_loss)):
            save_loss_2.append(save_test_loss[j])
        plt.plot(plt_x2, save_loss_1, 'b')
        plt.plot(plt_x2, save_loss_2, 'r')
        out2 = os.path.join(args.out_dir, args.folds, 'loss.png')
        plt.savefig(out2)
        plt.pause(1)
        plt.close()

        del model_ONE_feature, model_TWO_feature, model_class
    print(results_fold)
    print(cm_sum)
    with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
        f.write(str(cm_sum) + "\n")

    a = np.array(cm_sum)
    acc = calculate_all_prediction(a)
    k = kappa(cm_sum)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
        f.write("\n" + "acc:" + str(acc) + "\n")
        f.write("kappa:" + str(k) + "\n")

    MF_sum = 0
    for i in range(0, 5):
        with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
            PR = calculate_label_prediction(a, i)
            f.write(str(i) + ",PR:" + str(PR) + "\n")
            RC = calculate_label_recall(a, i)
            f.write(str(i) + ",RC:" + str(RC) + "\n")
            MF = calculate_f1(PR, RC)
            f.write(str(i) + ",MF:" + str(MF) + "\n")
            MF_sum += MF

    with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
        f.write("MF_sum:" + str(MF_sum / 5))
    return cm_sum


def calculate_all_prediction(confMatrix):

    total_sum = confMatrix.sum()
    print("总的样本数：", total_sum)
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100 * float(correct_sum) / float(total_sum), 2)
    return prediction


def calculate_label_prediction(confMatrix, labelidx):

    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return prediction


def calculate_label_recall(confMatrix, labelidx):

    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return recall


def calculate_f1(prediction, recall):
    if (prediction + recall) == 0:
        return 0
    return round(2 * prediction * recall / (prediction + recall), 2)


def kappa(confusion_matrix):
    confusion = confusion_matrix.cpu().numpy()
    pe_rows = np.sum(confusion, axis=0)
    pe_cols = np.sum(confusion, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion) / float(sum_total)
    return (po - pe) / (1 - pe)


if __name__ == '__main__':

    start_time = time.time()
    setup_seed(2023)
    cm_sum = main()
    a = np.array(cm_sum)
    acc = calculate_all_prediction(a)
    k = kappa(cm_sum)
    print("acc:", acc)
    print("kappa:", k)
    MF_sum = 0
    for i in range(0, 5):
        PR = calculate_label_prediction(a, i)
        print(str(i) + ",PR:" + str(PR))
        RC = calculate_label_recall(a, i)
        print(str(i) + ",RC:" + str(RC))
        MF = calculate_f1(PR, RC)
        print(str(i) + ",MF:" + str(MF))
        MF_sum += MF
        print("**********")
    print(MF_sum / 5)

    end_time = time.time()
    print("time:", end_time - start_time)
