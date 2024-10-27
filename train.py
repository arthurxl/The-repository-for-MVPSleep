import time
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score, classification_report
from data_augmentation import *


def train_feature(args, model_ONE_feature, model_TWO_feature, optimizer_ONE_feature, optimizer_TWO_feature,
                  train_loader, lossFunc_feature):
    print("******************train_feature******************")
    model_ONE_feature.train()
    model_TWO_feature.train()
    start_time = time.time()
    total_loss_ONE = 0
    total_loss_TWO = 0
    num = 0
    for step, (data_ONE1, lms1, lms2, label) in enumerate(train_loader):

        num += (data_ONE1.shape[0] * data_ONE1.shape[1])

        ##############################
        # data_ONE2 = gaussian_noise(data_ONE2)
        # data_ONE2 = time_reverse(data_ONE2)
        # data_ONE2 = sign_flip(data_ONE2)
        data_ONE1 = data_ONE1.cuda()
        mixup = MixUp(args, 0.4)
        data_ONE2 = mixup(data_ONE1)

        data_ONE2 = data_ONE2.cuda()
        lms1 = lms1.cuda()
        lms2 = lms2.cuda()

        x1_online_ONE, x1_target_ONE, x2_online_ONE, x2_target_ONE = model_ONE_feature(data_ONE1, data_ONE2)
        loss_ONE_1 = lossFunc_feature(x1_online_ONE, x2_target_ONE)
        loss_ONE_2 = lossFunc_feature(x1_target_ONE, x2_online_ONE)
        loss_ONE = loss_ONE_1 + loss_ONE_2

        x1_online_TWO, x1_target_TWO, x2_online_TWO, x2_target_TWO = model_TWO_feature(lms1, lms2)
        loss_TWO_1 = lossFunc_feature(x1_online_TWO, x2_target_TWO)
        loss_TWO_2 = lossFunc_feature(x1_target_TWO, x2_online_TWO)
        loss_TWO = loss_TWO_1 + loss_TWO_2

        optimizer_ONE_feature.zero_grad()
        model_ONE_feature.zero_grad()
        loss_ONE.backward()
        optimizer_ONE_feature.step()
        total_loss_ONE += loss_ONE.item()

        optimizer_TWO_feature.zero_grad()
        model_TWO_feature.zero_grad()
        loss_TWO.backward()
        optimizer_TWO_feature.step()
        total_loss_TWO += loss_TWO.item()

    print("train_feature_time:", time.time() - start_time)
    print("loss_ONE:", total_loss_ONE / num)
    print("loss_TWO:", total_loss_TWO / num)
    return (total_loss_ONE + total_loss_TWO) / (2 * num)


def train_class_epoch(args, train_loader, context_ONE_model, context_TWO_model,
                      model, optimizer_one, lossFunc_class):
    model.train()

    start = time.time()
    print("******************train_class******************")
    train_loss = 0
    correct = 0
    total = 0
    for step, (data_ONE1, lms1, _, label) in enumerate(train_loader):

        data_ONE1 = data_ONE1.cuda()
        lms1 = lms1.cuda()
        label = label.cuda()

        with torch.no_grad():
            x_online_ONE = context_ONE_model.test(data_ONE1)
            x_online_TWO = context_TWO_model.test(lms1)
            x_output_one = x_online_ONE.detach()
            x_output_two = x_online_TWO.detach()

        output = model(x_output_one, x_output_two)
        output = output.transpose(2, 1)
        loss, correct_batch, total_batch, _ = seq_cel(output, label, output.size(1))
        train_loss += loss.item()
        correct += correct_batch
        total += total_batch
        optimizer_one.zero_grad()
        loss.backward()
        optimizer_one.step()

    print("train_class_epoch:", time.time() - start)
    return correct / total, train_loss / total



def test_class_epoch(args, test_loader, context_ONE_model, context_TWO_model,
                     model, lossFunc_class):
    context_ONE_model.eval()
    context_TWO_model.eval()
    model.eval()

    start = time.time()
    print("******************test_class******************")
    # num = 0
    train_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    for i, (data_ONE1, lms1, _, label) in enumerate(test_loader):

        data_ONE1 = data_ONE1.cuda()
        lms1 = lms1.cuda()

        for j in range(0, len(label.reshape(-1))):
            y_true.append(label.reshape(-1)[j].cpu())
        label = label.cuda()


        with torch.no_grad():
            x_online_ONE = context_ONE_model.test(data_ONE1)
            x_online_TWO = context_TWO_model.test(lms1)
            x_output_one = x_online_ONE.detach()
            x_output_two = x_online_TWO.detach()

            output = model(x_output_one, x_output_two)
            output = output.transpose(2, 1)
            loss, correct_batch, total_batch, out = seq_cel(output, label, output.size(1))

            train_loss += loss.item()
            correct += correct_batch
            total += total_batch
            output_temp = out.reshape(-1)
            for j in range(0, len(output_temp.reshape(-1))):
                y_pred.append(output_temp.reshape(-1)[j].cpu())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    mf1 = f1_score(y_true, y_pred, average='macro')
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    k = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = classification_report(y_true, y_pred, digits=4)
    print("test_class_epoch:", time.time() - start)
    return correct / total, train_loss / total, mf1, per_class_f1, k, cm, per_class_acc

def seq_cel(pred, gt, class_num):
    # seq cross entropy loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, gt)
    total = torch.numel(gt)

    out = pred.max(1)[1]
    corr = torch.sum(torch.eq(out, gt)).item()
    return loss, corr, total, out