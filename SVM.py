import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from Dataset import SEEDIVDataset_indp, load_data_indp, reshape_input, load_data_indp_original, load_data_indp_original1
import copy
from models import ResNet18_, create_model, DANN, FeatureExtractor, Classifier, Discriminator, DANN_resnet
from tca import TCA
import wandb
# from models import create_model
import argparse
from DANN import train_DANN


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = "./checkpoint"


def normalize(train_data, test_data):
    m = train_data.mean(dim = 0, keepdim=True)
    s = train_data.std(dim=0, unbiased=True, keepdim=True)
    train_data = (train_data-m)/s

    m = test_data.mean(dim = 0, keepdim=True)
    s = test_data.std(dim=0, unbiased=True, keepdim=True)
    test_data = (test_data-m)/s

    return train_data, test_data

def test(model, test_dataloader, device, alpha=0):
    model.eval()
    n_correct = 0
    n_total = 0
    accuracy = 0
    target_iter = iter(test_dataloader)
    with torch.no_grad():
        for i in range(len(test_dataloader)):
            target_data = target_iter._next_data()
            t_img, t_label = target_data
            t_img = t_img.to(device)
            t_label = t_label.long().to(device)
            batch_size = len(t_label)

            class_output, _ = model(input_data=t_img, lamda=alpha)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size
        accuracy = float(n_correct) / n_total
        return accuracy


def test_resnet(model, test_dataloader, device, alpha=0):
    model.eval()
    n_correct = 0
    n_total = 0
    accuracy = 0
    target_iter = iter(test_dataloader)
    with torch.no_grad():
        for i in range(len(test_dataloader)):
            target_data = target_iter._next_data()
            t_img, t_label = target_data
            t_img = t_img.unsqueeze(1)
            t_img = reshape_input(t_img)
            t_img = t_img.to(device)
            t_label = t_label.long().to(device)
            batch_size = len(t_label)

            class_output, _ = model(input_data=t_img, lamda=alpha)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size
        accuracy = float(n_correct) / n_total
        return accuracy


def train_svm(args, train_dataset, test_dataset):
    validation_accs = np.zeros(15)
    for i in range(15):
        run = wandb.init(project=f'{args.model}_4_19',
                         name=f'fold:{i}',
                         config=args,
                         reinit=True)
        train_data = train_dataset[i]
        test_data = test_dataset[i]

        train_data.data = train_data.data.reshape(train_data.data.shape[0], -1)
        test_data.data = test_data.data.reshape(test_data.data.shape[0], -1)

        pca = PCA(n_components=19)
        pca.fit(train_data.data, train_data.label)
        train_data.data = pca.transform(train_data.data)
        test_data.data = pca.transform(test_data.data)

        # lda = LinearDiscriminantAnalysis(n_components=3)
        # lda.fit(train_data.data, train_data.label)
        # train_data.data = lda.transform(train_data.data)
        # test_data.data = lda.transform(test_data.data)

        clf = make_pipeline(StandardScaler(), SVC(C=args.svm_c, kernel=args.svm_kernel, gamma='auto'))
        clf.fit(train_data.data, train_data.label)
        test_predict = clf.predict(test_data.data)
        # print(classification_report(np.squeeze(test_dataset.y), test_predict))
        validation_accs[i] = accuracy_score(test_data.label, test_predict)
        print(f"Fold {i} acc: {validation_accs[i]:.4f}")
        wandb.log({'acc':validation_accs[i]})
        run.finish()
    run = wandb.init(project=f'{args.model}_4_19',
                     name=f'Final',
                     config=args,
                     reinit=True)
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")
    wandb.log({'acc_mean':acc_mean, 'acc_std':acc_std})
    os.makedirs('results', exist_ok=True)
    np.save('results/svm', validation_accs)