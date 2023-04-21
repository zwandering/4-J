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
from ResNet import train_resnet
from ADDA import train_ADDA
from implementation_PR_PL import train_prpl

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


def train_generalization(args, train_dataset_all, test_dataset_all):
    validation_accs = np.zeros(15)
    for idx in range(15):
        run = wandb.init(project=f'{args.model}_4_19',
                   name = f'fold:{idx}',
                   config=args,
                   reinit=True)
        train_dataset = train_dataset_all[idx]
        test_dataset = test_dataset_all[idx]
        train_dataset.data = train_dataset.data.reshape(train_dataset.data.shape[0], -1)
        test_dataset.data = test_dataset.data.reshape(test_dataset.data.shape[0], -1)

        # pca = PCA(n_components=50)
        #
        # pca.fit(train_dataset.data, train_dataset.label)
        # train_dataset.data, test_dataset.data = torch.tensor(pca.transform(train_dataset.data)).float(), torch.tensor(pca.transform(test_dataset.data)).float()

        train_dataset.data, test_dataset.data = normalize(train_dataset.data, test_dataset.data)

        test_x, test_y = torch.tensor(test_dataset.data).to(device), torch.tensor(test_dataset.label).to(device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

        model = create_model(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_acc = 0.
        for epoch in range(args.num_epoch):
            model.train()
            total_loss = 0.
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                loss = model.compute_loss((x.float(), y))

                loss.backward()
                optimizer.step()
                total_loss += loss.detach().cpu().numpy()
            if epoch % args.display_epoch == 0:
                model.eval()
                # _, test_y_pred = torch.max(model(test_x.float()), dim=1)
                train_acc = test(model=model, test_dataloader=train_loader, device=device)
                test_acc = test(model=model, test_dataloader=test_loader, device=device)
                if test_acc > best_acc:
                    best_acc = test_acc
                    # filename = f"{args.model}_checkpoint.pt"
                    # os.makedirs(checkpoint_dir, exist_ok=True)
                    # file_path = os.path.join(checkpoint_dir, filename)
                    # torch.save(model.state_dict(), file_path)
                print(f"Epoch {epoch}, Loss {total_loss / len(train_loader):.4f}, Train Acc {train_acc:.4f} Test Acc {test_acc:.4f}")
                wandb.log({'Epoch': epoch, 'loss': total_loss / len(train_loader), 'Train Acc': train_acc, 'Test Acc': test_acc})
        validation_accs[idx] = best_acc
        wandb.log({'best acc':best_acc})
        run.finish()
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")
    run = wandb.init(project=f'{args.model}_4_19',
                     name='Final',
                     config=args,
                     reinit=True)
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")
    wandb.log({'acc_mean':acc_mean, 'acc_std': acc_std})
    run.finish()


def train_generalization_tca(args, train_dataset_all, test_dataset_all):
    validation_accs = np.zeros(15)
    for idx in range(15):
        train_dataset = train_dataset_all[idx]
        test_dataset = test_dataset_all[idx]
        train_dataset.data = train_dataset.data.reshape(train_dataset.data.shape[0], -1)
        test_dataset.data = test_dataset.data.reshape(test_dataset.data.shape[0], -1)
        pca =PCA(n_components=20)
        tca = TCA(kernel_type='rbf', dim=310)

        pca.fit(train_dataset.data, train_dataset.label)
        train_dataset.data, test_dataset.data = torch.tensor(pca.transform(train_dataset.data)).float(), torch.tensor(pca.transform(test_dataset.data)).float()

        print('PCA Complete')
        train_dataset.data, test_dataset.data = tca.fit(train_dataset.data, test_dataset.data)

        train_dataset.data, test_dataset.data = normalize(train_dataset.data, test_dataset.data)

        test_x, test_y = torch.tensor(test_dataset.data).to(device), torch.tensor(test_dataset.label).to(device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

        model = create_model(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_acc = 0.
        for epoch in range(args.num_epoch):
            model.train()
            total_loss = 0.
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                loss = model.compute_loss((x, y))

                loss.backward()
                optimizer.step()
                total_loss += loss.detach().cpu().numpy()
            if epoch % args.display_epoch == 0:
                model.eval()
                # _, test_y_pred = torch.max(model(test_x.float()), dim=1)
                train_acc = test(model=model, test_dataloader=train_loader, device=device)
                test_acc = test(model=model, test_dataloader=test_loader, device=device)
                if test_acc > best_acc:
                    best_acc = test_acc
                    # filename = f"{args.model}_checkpoint.pt"
                    # os.makedirs(checkpoint_dir, exist_ok=True)
                    # file_path = os.path.join(checkpoint_dir, filename)
                    # torch.save(model.state_dict(), file_path)
                print(f"Epoch {epoch}, Loss {total_loss / len(train_loader):.4f}, Train Acc {train_acc:.4f} Test Acc {test_acc:.4f}")
        validation_accs[idx] = best_acc
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")


def test_adda(feature_extractor, classifier, test_dataloader):
    feature_extractor.eval()
    classifier.eval()
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

            class_output = classifier(feature_extractor(t_img))
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size
        accuracy = float(n_correct) / n_total
        return accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="General Traning Pipeline")

    parser.add_argument("--model", type=str, default='SVM')
    parser.add_argument("--svm_c", type=float, default=0.001)
    parser.add_argument("--svm_kernel", choices=['linear', 'rbf', 'sigmoid', 'poly'], default='linear')
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--triplet_weight", type=float, default=.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--display_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--penalty_weight", type=float, default=.5)
    parser.add_argument("--weight_decay", type=float, default=.0)
    parser.add_argument("--variance_weight", type=float, default=.5)
    parser.add_argument("--wgan_lamda", type=float, default=10.)
    parser.add_argument("--pretrain_epoch", type=int, default=20)
    parser.add_argument("--advtrain_iteration", type=int, default=10000)
    parser.add_argument("--critic_iters", type=int, default=10)
    parser.add_argument("--gen_iters", type=int, default=10000)
    parser.add_argument("--is_augmentation", action='store_true')
    parser.add_argument("--display_iters", type=int, default=10)

    args = parser.parse_args()

    # fix random seed for reproducibility
    if args.seed != None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # train_dataset, test_dataset = load_data_indp('SEED-IV')
    train_dataset, test_dataset = load_data_indp_original1('SEED-IV')


    print('Load Data Success!')

    if args.model == 'SVM':
        train_svm(args, train_dataset, test_dataset)
    elif args.model == 'resnet':
        train_resnet(args, train_dataset, test_dataset)
    elif args.model == 'DANN':
        train_DANN(args, train_dataset, test_dataset)
    elif args.model == 'ADDA':
        train_ADDA(args, train_dataset, test_dataset)
    elif args.model == 'MLP' or args.model == 'IRM':
        train_generalization(args, train_dataset, test_dataset)
    elif args.model == 'tca':
        train_generalization_tca(args, train_dataset, test_dataset)
    elif args.model == 'prpl':
        train_prpl()
    else:
        raise ValueError("Unknown model type!")
