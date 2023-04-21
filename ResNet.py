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
from SVM import train_svm


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

            class_output = model(t_img)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size
        accuracy = float(n_correct) / n_total
        return accuracy


def train_resnet(args, train_dataset, test_dataset):
    num_epochs = args.num_epoch
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_acc = []
    best_models = []
    for i in range(15):
        run = wandb.init(project='ResNet_4_20',
                         name=f'fold:{i}',
                         config=args,
                         reinit=True)
        # record best acc
        best_acc = 0.0
        # ResNet18 58.23
        model = ResNet18_()
        # model = ResNet50_()
        # model = CNN2().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.4, last_epoch=-1)
        train_data = train_dataset[i]
        test_data = test_dataset[i]
        train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=8, drop_last=False)
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (data, labels) in enumerate(train_dataloader):
                data = data.unsqueeze(1)
                data = reshape_input(data)
                # labels = labels.unsqueeze(1)
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(data)
                # print(output.shape, labels.shape)
                loss = criterion(output, labels)
                # print(loss.detach())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            scheduler.step()
            # Evaluate on test set after each epoch
            test_dataloader = DataLoader(test_data, batch_size=200, shuffle=False, drop_last=False)
            with torch.no_grad():
                model.eval()
                correct = 0
                total = 0
                train_acc = test_resnet(model, train_dataloader, device)
                accuracy = test_resnet(model, test_dataloader, device)
                # for data, labels in test_dataloader:
                #     data = data.unsqueeze(1)
                #     data = reshape_input(data)
                #     data, labels = data.to(device), labels.to(device)
                #     output = model(data)
                #     _, predicted = torch.max(output.data, 1)
                #     # print(predicted, labels)
                #     total += labels.size(0)
                #     # print(predicted, labels)
                #     correct += (predicted == labels).sum().item()
                #
                # accuracy = 100 * correct / total
                if accuracy > best_acc:
                    best_acc = accuracy
                    # Save the model with the highest accuracy
                    best_model = copy.deepcopy(model.state_dict())
                print(f"Subject {i + 1}, Epoch {epoch + 1}: Train Loss {train_loss:.6f}, Train accuracy {train_acc:.4f} Accuracy {accuracy:.4f}")
                wandb.log({'Epoch':epoch, 'Train Loss':train_loss,'Train Accuracy':train_acc, 'Accuracy':accuracy})
        print(best_acc)
        wandb.log({'best acc': best_acc})
        total_acc.append(best_acc)
        best_models.append(best_model)
        run.finish()
    run = wandb.init(project='ResNet_4_20',
                     name=f'fold:{i}',
                     config=args,
                     reinit=True)
    total_acc = np.asarray(total_acc)
    acc = np.mean(total_acc)
    std = np.std(total_acc)
    wandb.log({'acc mean':acc, 'acc std':std})
    # acc = sum(total_acc) / len(total_acc)
    print(f"Total Accuracy {acc:.4f}% Std {std:.4f}")
    run.finish()


    # os.makedirs(f"indp_acc_{acc:.2f}", exist_ok=True)
    # for i in range(15):
    #     torch.save(best_models[i], f'indp_acc_{acc:.2f}/model_{i:02d}.pt')