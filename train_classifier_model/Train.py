from DataSet import DataSet
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from model import SE_VGG
from tqdm import tqdm

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    # ============================ step 1/5 数据 ============================
    train_data_path = r'C:\Users\NailinLiao\PycharmProjects\DeepWay\Data\train'
    val_data_path = r'C:\Users\NailinLiao\PycharmProjects\DeepWay\Data\test'
    num_classes = 2

    MAX_EPOCH = 3
    BATCH_SIZE = 4  # 16 / 8 / 4 / 2 / 1
    LR = 0.001
    log_interval = 2
    val_interval = 1
    classes = 2
    start_epoch = -1
    lr_decay_step = 1

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = DataSet(train_data_path, (224, 224), transform)
    train_data_loadr = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

    val_set = DataSet(val_data_path, (224, 224), transform)
    valid_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE)
    # ============================ step 2/5 模型 ============================
    model = SE_VGG(2)
    model.to(device)
    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 选择优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # 设置学习率下降策略

    # ============================ step 5/5 训练 ============================
    train_curve = list()
    valid_curve = list()

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        model.train()
        for i, data in enumerate(train_data_loadr):

            # forward
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} lr:{}".format(
                    epoch, MAX_EPOCH, i + 1, len(train_data_loadr), loss_mean, correct / total,
                    scheduler.get_last_lr()))
                loss_mean = 0.
        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # bs, ncrops, c, h, w = inputs.size()
                    #
                    #
                    # outputs = model(inputs.view(-1, c, h, w))
                    # outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                    #
                    # loss = criterion(outputs_avg, labels)
                    #
                    # _, predicted = torch.max(outputs_avg.DataSet, 1)

                    outputs = model(inputs)

                    # backward
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                loss_val_mean = loss_val / len(valid_loader)
                valid_curve.append(loss_val_mean)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val))
            model.train()

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_data_loadr)
    valid_x = np.arange(1,
                        len(valid_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.savefig(os.path.join('./', "loss_curve.png"))
