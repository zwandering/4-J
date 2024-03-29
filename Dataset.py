import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

device  = 'cuda'


class SEEDIVDataset_indp(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y


# without norm
def load_data_indp(path='SEED-IV'):
    if not os.path.exists("./data/SEED-IV_concatenate_unfold/"):
        make_datasets_unfold(path='SEED-IV')

    train_data_loader, test_data_loader = [], []
    for i in range(15):
        train_data = np.load(f'/home/huteng/zhuhaokun/4-J/proj/data/SEED-IV_concatenate_unfold/{i}/train_data.npy')
        train_label = np.load(f'/home/huteng/zhuhaokun/4-J/proj/data/SEED-IV_concatenate_unfold/{i}/train_label.npy')
        test_data = np.load(f'/home/huteng/zhuhaokun/4-J/proj/data/SEED-IV_concatenate_unfold/{i}/test_data.npy')
        test_label = np.load(f'/home/huteng/zhuhaokun/4-J/proj/data/SEED-IV_concatenate_unfold/{i}/test_label.npy')
        train_data_loader.append(SEEDIVDataset_indp(train_data, train_label))
        test_data_loader.append(SEEDIVDataset_indp(test_data, test_label))

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

        # train_data1_ = np.concatenate(test_data[0:i] + test_data[i + 1:15], axis=0)
        # train_data2_ = np.concatenate(test_data[15:i + 15] + test_data[i + 16:30], axis=0)
        # train_data3_ = np.concatenate(test_data[30:i + 30] + test_data[i + 31:45], axis=0)

        # train_data_tmp = [train_data1, train_data2, train_data3, train_data1_, train_data2_, train_data3_]
        train_data_tmp = [train_data1, train_data2, train_data3, ]
        train_data_final = np.concatenate(train_data_tmp, axis=0)

        train_label1 = np.concatenate(train_label[:i] + train_label[i + 1:15], axis=0)
        train_label2 = np.concatenate(train_label[15:i + 15] + train_label[i + 16:30], axis=0)
        train_label3 = np.concatenate(train_label[30:i + 30] + train_label[i + 31:45], axis=0)

        # train_label1_ = np.concatenate(test_label[:i] + test_label[i + 1:15], axis=0)
        # train_label2_ = np.concatenate(test_label[15:i + 15] + test_label[i + 16:30], axis=0)
        # train_label3_ = np.concatenate(test_label[30:i + 30] + test_label[i + 31:45], axis=0)

        # train_label_tmp = [train_label1, train_label2, train_label3, train_label1_, train_label2_, train_label3_]
        train_label_tmp = [train_label1, train_label2, train_label3, ]
        train_label_final = np.concatenate(train_label_tmp, axis=0)


        test_data1 = np.concatenate(train_data[i:i + 1], axis=0)
        test_data2 = np.concatenate(train_data[i + 15:i + 16], axis=0)
        test_data3 = np.concatenate(train_data[i + 30:i + 31], axis=0)

        # test_data1_ = np.concatenate(test_data[i:i + 1], axis=0)
        # test_data2_ = np.concatenate(test_data[i + 15:i + 16], axis=0)
        # test_data3_ = np.concatenate(test_data[i + 30:i + 31], axis=0)

        # test_data_tmp = [test_data1, test_data2, test_data3, test_data1_, test_data2_, test_data3_]
        test_data_tmp = [test_data1, test_data2, test_data3, ]
        test_data_final = np.concatenate(test_data_tmp, axis=0)

        test_label1 = np.concatenate(train_label[i:i + 1], axis=0)
        test_label2 = np.concatenate(train_label[i + 15:i + 16], axis=0)
        test_label3 = np.concatenate(train_label[i + 30:i + 31], axis=0)

        # test_label1_ = np.concatenate(test_label[i:i + 1], axis=0)
        # test_label2_ = np.concatenate(test_label[i + 15:i + 16], axis=0)
        # test_label3_ = np.concatenate(test_label[i + 30:i + 31], axis=0)

        test_label_tmp = [test_label1, test_label2, test_label3, test_label1_, test_label2_, test_label3_]
        test_label_tmp = [test_label1, test_label2, test_label3, ]
        test_label_final = np.concatenate(test_label_tmp, axis=0)

        train_data_loader.append(SEEDIVDataset_indp(train_data_final, train_label_final))
        test_data_loader.append(SEEDIVDataset_indp(test_data_final, test_label_final))

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


def get_data(data):
    return data["train_data"], data["train_label"], data["test_data"], data["test_label"]


def make_datasets_unfold(path='SEED-IV'):
    data = []
    for file1 in os.listdir(path):
        file1 = os.path.join(path, file1)
        path_list = os.listdir(file1)
        # path_list.sort(key=lambda x:int(x))
        # print(path_list)
        experiment_data_list = []
        for file2 in path_list:
            session_path = os.path.join(file1, file2)
            # print(file2)
            session_data = {
                "train_data": np.load(session_path + "/train_data.npy"),
                "train_label": np.load(session_path + "/train_label.npy"),
                "test_data": np.load(session_path + "/test_data.npy"),
                "test_label": np.load(session_path + "/test_label.npy")
            }
            experiment_data_list.append(session_data)
        data.append(experiment_data_list)

    test_data_lists = []
    test_label_lists = []
    train_data_lists = []
    train_label_lists = []

    for test_session_id in range(len(data[0])):
        dataset_path = "./data/SEED-IV_concatenate_unfold/" + str(test_session_id) + "/"
        if os.path.exists(dataset_path) == False:
            os.makedirs(dataset_path)

        test_data_list = []
        test_label_list = []
        train_data_list = []
        train_label_list = []
        for experiment_id in range(len(data)):
            raw_train_data, raw_train_label, raw_test_data, raw_test_label = get_data(
                data[experiment_id][test_session_id])

            raw_list = []
            for i in range(len(raw_train_data)):
                raw_list.append(raw_train_data[i].flatten())
            raw_list = np.array(raw_list)
            raw_train_data = raw_list

            test_data_list.append(raw_train_data)
            test_label_list.append(raw_train_label)
            for session_id in range(len(data[experiment_id])):
                if session_id == test_session_id:
                    continue
                else:
                    raw_train_data, raw_train_label, raw_test_data, raw_test_label = get_data(
                        data[experiment_id][session_id])

                    raw_list = []
                    for i in range(len(raw_train_data)):
                        raw_list.append(raw_train_data[i].flatten())
                    raw_list = np.array(raw_list)
                    raw_train_data = raw_list

                    train_data_list.append(raw_train_data)
                    train_label_list.append(raw_train_label)

        test_data_list = np.concatenate(test_data_list)
        test_label_list = np.concatenate(test_label_list)
        test_data_lists.append(test_data_list)
        test_label_lists.append(test_label_list)
        train_data_list = np.concatenate(train_data_list)
        train_label_list = np.concatenate(train_label_list)
        train_data_lists.append(train_data_list)
        train_label_lists.append(train_label_list)

        train_data_array = np.array(train_data_list)
        test_data_array = np.array(test_data_list)
        train_label_array = np.array(train_label_list)
        test_label_array = np.array(test_label_list)
        np.save(dataset_path + "train_data.npy", train_data_array)
        np.save(dataset_path + "test_data.npy", test_data_array)
        np.save(dataset_path + "train_label.npy", train_label_array)
        np.save(dataset_path + "test_label.npy", test_label_array)


def load_data_indp_original(path='SEED-IV'):
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

        # train_data1_ = np.concatenate(test_data[0:i] + test_data[i + 1:15], axis=0)
        # train_data2_ = np.concatenate(test_data[15:i + 15] + test_data[i + 16:30], axis=0)
        # train_data3_ = np.concatenate(test_data[30:i + 30] + test_data[i + 31:45], axis=0)

        # train_data_tmp = [train_data1, train_data2, train_data3, train_data1_, train_data2_, train_data3_]
        train_data_tmp = [train_data1, train_data2, train_data3,]
        train_data_final = np.concatenate(train_data_tmp, axis=0)

        train_label1 = np.concatenate(train_label[:i] + train_label[i + 1:15], axis=0)
        train_label2 = np.concatenate(train_label[15:i + 15] + train_label[i + 16:30], axis=0)
        train_label3 = np.concatenate(train_label[30:i + 30] + train_label[i + 31:45], axis=0)

        # train_label1_ = np.concatenate(test_label[:i] + test_label[i + 1:15], axis=0)
        # train_label2_ = np.concatenate(test_label[15:i + 15] + test_label[i + 16:30], axis=0)
        # train_label3_ = np.concatenate(test_label[30:i + 30] + test_label[i + 31:45], axis=0)

        # train_label_tmp = [train_label1, train_label2, train_label3, train_label1_, train_label2_, train_label3_]
        train_label_tmp = [train_label1, train_label2, train_label3,]
        train_label_final = np.concatenate(train_label_tmp, axis=0)

        test_data1 = np.concatenate(train_data[i:i + 1], axis=0)
        test_data2 = np.concatenate(train_data[i + 15:i + 16], axis=0)
        test_data3 = np.concatenate(train_data[i + 30:i + 31], axis=0)

        # test_data1_ = np.concatenate(test_data[i:i + 1], axis=0)
        # test_data2_ = np.concatenate(test_data[i + 15:i + 16], axis=0)
        # test_data3_ = np.concatenate(test_data[i + 30:i + 31], axis=0)

        # test_data_tmp = [test_data1, test_data2, test_data3, test_data1_, test_data2_, test_data3_]
        test_data_tmp = [test_data1, test_data2, test_data3,]
        test_data_final = np.concatenate(test_data_tmp, axis=0)

        test_label1 = np.concatenate(train_label[i:i + 1], axis=0)
        test_label2 = np.concatenate(train_label[i + 15:i + 16], axis=0)
        test_label3 = np.concatenate(train_label[i + 30:i + 31], axis=0)

        # test_label1_ = np.concatenate(test_label[i:i + 1], axis=0)
        # test_label2_ = np.concatenate(test_label[i + 15:i + 16], axis=0)
        # test_label3_ = np.concatenate(test_label[i + 30:i + 31], axis=0)

        # test_label_tmp = [test_label1, test_label2, test_label3, test_label1_, test_label2_, test_label3_]
        test_label_tmp = [test_label1, test_label2, test_label3,]
        test_label_final = np.concatenate(test_label_tmp, axis=0)

        train_data_loader.append(SEEDIVDataset_indp(train_data_final, train_label_final))
        test_data_loader.append(SEEDIVDataset_indp(test_data_final, test_label_final))

    return train_data_loader, test_data_loader


def load_data_indp_original1(path='SEED-IV'):
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



# if __name__=="__main__":
    # train_data_loader, test_data_loader = load_data_indp()
    # train_data_loader1, test_data_loader1 = tmp()
    # for i in range(15):
    #     train_data_loader[i].data = train_data_loader[i].data.reshape(train_data_loader[i].data.shape[0],-1)
    #     # print(train_data_loader[i].data.shape)
    #     # print(train_data_loader1[i].data.shape)
    #     for j in range(train_data_loader[i].data.shape[0]):
    #         for k in range(train_data_loader[i].data.shape[-1]):
    #             if not train_data_loader[i].data[j,k] == train_data_loader1[i].data[j,k]:
    #                 # print(train_data_loader[i].data[j,k], train_data_loader1[i].data[j,k])
    #                 print(i, j, k)
    #         # if not train_data_loader[i].data[j] == train_data_loader1[i].data[j]:
    #         #     print('not equal')
    #     # print(test_data_loader[i].data.shape)
    #     # print(test_data_loader1[i].data.shape)