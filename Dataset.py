import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

device  = 'cuda'


class SEEDIVDataset_indp(Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y


# without norm
def load_data_indp(path='SEED-IV'):
    train_data, train_label, test_data, test_label = [], [], [], []
    for file1 in os.listdir(path):
        file1 = os.path.join(path, file1)
        for file2 in os.listdir(file1):
            file2 = os.path.join(file1, file2)
            train_data.append(np.load(os.path.join(file2, 'train_data.npy')))
            train_label.append(np.load(os.path.join(file2, 'train_label.npy')))
            test_data.append(np.load(os.path.join(file2, 'test_data.npy')))
            test_label.append(np.load(os.path.join(file2, 'test_label.npy')))

    train_data_loader, test_data_loader = [], []
    for i in range(15):
        train_data1 = np.concatenate(train_data[0:i] + train_data[i + 1:15], axis=0)
        train_data2 = np.concatenate(train_data[15:i + 15] + train_data[i + 16:30], axis=0)
        train_data3 = np.concatenate(train_data[30:i + 30] + train_data[i + 31:45], axis=0)

        train_data1_ = np.concatenate(test_data[0:i] + test_data[i + 1:15], axis=0)
        train_data2_ = np.concatenate(test_data[15:i + 15] + test_data[i + 16:30], axis=0)
        train_data3_ = np.concatenate(test_data[30:i + 30] + test_data[i + 31:45], axis=0)

        train_data_tmp = [train_data1, train_data2, train_data3, train_data1_, train_data2_, train_data3_]
        train_data_final = np.concatenate(train_data_tmp, axis=0)

        train_label1 = np.concatenate(train_label[:i] + train_label[i + 1:15], axis=0)
        train_label2 = np.concatenate(train_label[15:i + 15] + train_label[i + 16:30], axis=0)
        train_label3 = np.concatenate(train_label[30:i + 30] + train_label[i + 31:45], axis=0)

        train_label1_ = np.concatenate(test_label[:i] + test_label[i + 1:15], axis=0)
        train_label2_ = np.concatenate(test_label[15:i + 15] + test_label[i + 16:30], axis=0)
        train_label3_ = np.concatenate(test_label[30:i + 30] + test_label[i + 31:45], axis=0)

        train_label_tmp = [train_label1, train_label2, train_label3, train_label1_, train_label2_, train_label3_]
        train_label_final = np.concatenate(train_label_tmp, axis=0)

        test_data1 = np.concatenate(train_data[i:i + 1], axis=0)
        test_data2 = np.concatenate(train_data[i + 15:i + 16], axis=0)
        test_data3 = np.concatenate(train_data[i + 30:i + 31], axis=0)

        test_data1_ = np.concatenate(test_data[i:i + 1], axis=0)
        test_data2_ = np.concatenate(test_data[i + 15:i + 16], axis=0)
        test_data3_ = np.concatenate(test_data[i + 30:i + 31], axis=0)

        test_data_tmp = [test_data1, test_data2, test_data3, test_data1_, test_data2_, test_data3_]
        test_data_final = np.concatenate(test_data_tmp, axis=0)

        test_label1 = np.concatenate(train_label[i:i + 1], axis=0)
        test_label2 = np.concatenate(train_label[i + 15:i + 16], axis=0)
        test_label3 = np.concatenate(train_label[i + 30:i + 31], axis=0)

        test_label1_ = np.concatenate(test_label[i:i + 1], axis=0)
        test_label2_ = np.concatenate(test_label[i + 15:i + 16], axis=0)
        test_label3_ = np.concatenate(test_label[i + 30:i + 31], axis=0)

        test_label_tmp = [test_label1, test_label2, test_label3, test_label1_, test_label2_, test_label3_]
        test_label_final = np.concatenate(test_label_tmp, axis=0)

        train_data_loader.append(SEEDIVDataset_indp(train_data_final, train_label_final))
        test_data_loader.append(SEEDIVDataset_indp(test_data_final, test_label_final))

    return train_data_loader, test_data_loader


# try to add normalization
def load_data_indp_norm(path='SEED-IV'):
    train_data, train_label, test_data, test_label = [], [], [], []
    for file1 in os.listdir(path):
        file1 = os.path.join(path, file1)
        for file2 in os.listdir(file1):
            file2 = os.path.join(file1, file2)
            train_data_tmp = np.load(os.path.join(file2, 'train_data.npy'))
            train_label_tmp = np.load(os.path.join(file2, 'train_label.npy'))
            test_data_tmp = np.load(os.path.join(file2, 'test_data.npy'))
            test_label_tmp = np.load(os.path.join(file2, 'test_label.npy'))

            mean = np.mean(train_data_tmp)
            std = np.std(train_data_tmp)
            train_data_tmp = (train_data_tmp - mean) / std
            test_data_tmp = (test_data_tmp - mean) / std

            train_data.append(train_data_tmp)
            train_label.append(train_label_tmp)
            test_data.append(test_data_tmp)
            test_label.append(test_label_tmp)

            # train_data.append(np.load(os.path.join(file2, 'train_data.npy')))
            # train_label.append(np.load(os.path.join(file2, 'train_label.npy')))
            # test_data.append(np.load(os.path.join(file2, 'test_data.npy')))
            # test_label.append(np.load(os.path.join(file2, 'test_label.npy')))

    train_data_loader, test_data_loader = [], []
    for i in range(15):
        train_data1 = np.concatenate(train_data[0:i] + train_data[i + 1:15], axis=0)
        train_data2 = np.concatenate(train_data[15:i + 15] + train_data[i + 16:30], axis=0)
        train_data3 = np.concatenate(train_data[30:i + 30] + train_data[i + 31:45], axis=0)

        train_data1_ = np.concatenate(test_data[0:i] + test_data[i + 1:15], axis=0)
        train_data2_ = np.concatenate(test_data[15:i + 15] + test_data[i + 16:30], axis=0)
        train_data3_ = np.concatenate(test_data[30:i + 30] + test_data[i + 31:45], axis=0)

        train_data_tmp = [train_data1, train_data2, train_data3, train_data1_, train_data2_, train_data3_]
        train_data_final = np.concatenate(train_data_tmp, axis=0)

        train_label1 = np.concatenate(train_label[:i] + train_label[i + 1:15], axis=0)
        train_label2 = np.concatenate(train_label[15:i + 15] + train_label[i + 16:30], axis=0)
        train_label3 = np.concatenate(train_label[30:i + 30] + train_label[i + 31:45], axis=0)

        train_label1_ = np.concatenate(test_label[:i] + test_label[i + 1:15], axis=0)
        train_label2_ = np.concatenate(test_label[15:i + 15] + test_label[i + 16:30], axis=0)
        train_label3_ = np.concatenate(test_label[30:i + 30] + test_label[i + 31:45], axis=0)

        train_label_tmp = [train_label1, train_label2, train_label3, train_label1_, train_label2_, train_label3_]
        train_label_final = np.concatenate(train_label_tmp, axis=0)

        test_data1 = np.concatenate(train_data[i:i + 1], axis=0)
        test_data2 = np.concatenate(train_data[i + 15:i + 16], axis=0)
        test_data3 = np.concatenate(train_data[i + 30:i + 31], axis=0)

        test_data1_ = np.concatenate(test_data[i:i + 1], axis=0)
        test_data2_ = np.concatenate(test_data[i + 15:i + 16], axis=0)
        test_data3_ = np.concatenate(test_data[i + 30:i + 31], axis=0)

        test_data_tmp = [test_data1, test_data2, test_data3, test_data1_, test_data2_, test_data3_]
        test_data_final = np.concatenate(test_data_tmp, axis=0)

        test_label1 = np.concatenate(train_label[i:i + 1], axis=0)
        test_label2 = np.concatenate(train_label[i + 15:i + 16], axis=0)
        test_label3 = np.concatenate(train_label[i + 30:i + 31], axis=0)

        test_label1_ = np.concatenate(test_label[i:i + 1], axis=0)
        test_label2_ = np.concatenate(test_label[i + 15:i + 16], axis=0)
        test_label3_ = np.concatenate(test_label[i + 30:i + 31], axis=0)

        test_label_tmp = [test_label1, test_label2, test_label3, test_label1_, test_label2_, test_label3_]
        test_label_final = np.concatenate(test_label_tmp, axis=0)

        train_data_loader.append(SEEDIVDataset_indp(train_data_final, train_label_final))
        test_data_loader.append(SEEDIVDataset_indp(test_data_final, test_label_final))

    return train_data_loader, test_data_loader


def tmp():
    train_data_loader, test_data_loader = [], []
    for i in range(15):
        train_data = np.load(f'/home/huteng/zhuhaokun/4-J/proj/data/SEED-IV_concatenate_unfold/{i}/train_data.npy')
        train_label = np.load(f'/home/huteng/zhuhaokun/4-J/proj/data/SEED-IV_concatenate_unfold/{i}/train_label.npy')
        test_data = np.load(f'/home/huteng/zhuhaokun/4-J/proj/data/SEED-IV_concatenate_unfold/{i}/test_data.npy')
        test_label = np.load(f'/home/huteng/zhuhaokun/4-J/proj/data/SEED-IV_concatenate_unfold/{i}/test_label.npy')
        train_data_loader.append(SEEDIVDataset_indp(train_data, train_label))
        test_data_loader.append(SEEDIVDataset_indp(test_data, test_label))

    return train_data_loader, test_data_loader


def reshape_input(x):
    # x:[args.batch_size, 1, 62, 5]
    input = torch.zeros(x.shape[0], 5, 8, 9)
    for i in range(x.shape[0]):
        # 第一行
        for j in range(5):
            input[i, 0:5, 0, j+2] = x[i, 0, j, 0:5]
        # 2-6行
        for j in range(1,6):
            for k in range(9):
                input[i, 0:5, j, k] = x[i, 0, 9*(j-1)+5+k, 0:5]
        # 第七行
        for j in range(7):
            input[i, 0:5, 6, j + 1] = x[i, 0, j + 50, 0:5]
        #第八行
        for j in range(5):
            input[i, 0:5, 7, j + 2] = x[i, 0, j+57, 0:5]
    #print(input.shape)
    return input