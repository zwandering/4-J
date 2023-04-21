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


def train_DANN(args, train_dataset_all, test_datastet_all):
    validation_accs = np.zeros(15)
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    for idx in range(15):
        run = wandb.init(project='DANN_4_18_2',
                   name = f'fold:{idx}',
                   config=args,
                   reinit=True)

        train_dataset = train_dataset_all[idx]
        test_dataset = test_datastet_all[idx]
        train_dataset.data = train_dataset.data.reshape(train_dataset.data.shape[0], -1)
        test_dataset.data = test_dataset.data.reshape(test_dataset.data.shape[0], -1)

        train_dataset.data, test_dataset.data = normalize(train_dataset.data, test_dataset.data)
        # print(train_dataset.data.shape)
        # print(train_dataset.data[0])

        test_x, test_y = test_dataset.data.to(device), test_dataset.label.to(device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

        model = create_model(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.3, last_epoch=-1)

        train_iter = iter(train_loader)
        best_acc = 0.
        for epoch in range(args.num_epoch):
            model.train()

            len_dataloader = min(len(train_loader), len(test_loader))
            total_class_loss = 0.
            total_domain_loss = 0.
            for i, data in enumerate(test_loader):
                p = float(i + epoch * len_dataloader) / args.num_epoch / len_dataloader
                lamda = 0.5 * (2. / (1. + np.exp(-10 * p)) - 1)

                try:
                    input_s, class_labels = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    input_s, class_labels = next(train_iter)
                input_s, class_labels = input_s.to(device), class_labels.to(device)
                domain_source_labels = torch.zeros(len(input_s)).long().to(device)
                train_data = input_s.to(device).float(), class_labels.to(device), domain_source_labels.to(device)

                input_t, _ = data
                input_t = input_t.to(device)
                domain_target_labels = torch.ones(len(input_t)).long().to(device)
                test_data = input_t.to(device).float(), domain_target_labels.to(device)

                optimizer.zero_grad()

                # pred_class_label, pred_domain_label = model.forward(input_s, lamda)
                # class_loss = criterion(pred_class_label, class_labels)
                # domain_source_loss = criterion(pred_domain_label, domain_source_labels)
                #
                # _, pred_domain_label = model.forward(input_t, lamda)
                # domain_target_loss = criterion(pred_domain_label, domain_target_labels)
                #
                # domain_loss = domain_source_loss + domain_target_loss
                class_loss, domain_loss = model.compute_loss(train_data, test_data, lamda)
                loss = class_loss + domain_loss

                # class_loss, domain_loss = model.compute_loss(train_data, test_data, lamda)
                # loss = class_loss + domain_loss

                loss.backward()
                optimizer.step()


                total_class_loss += class_loss.detach().cpu().numpy()
                total_domain_loss += domain_loss.detach().cpu().numpy()

            if epoch % args.display_epoch == 0:
                with torch.no_grad():
                    model.eval()

                    train_acc = test(model=model, test_dataloader=train_loader, device=device, alpha=lamda)
                    test_acc = test(model=model, test_dataloader=test_loader, device=device, alpha=lamda)

                    # pred_class_label, _ = model(test_x.float(), lamda)
                    # _, test_y_pred = torch.max(pred_class_label, dim=1)
                    # test_acc = (test_y_pred == test_y).sum().item() / len(test_dataset)
                    if test_acc > best_acc:
                        best_acc = test_acc
                        filename = f"{args.model}_checkpoint.pt"
                        torch.save(model.state_dict(), os.path.join(checkpoint_dir, filename))
                    print(f"Epoch {epoch}, Class Loss {total_class_loss / len(train_loader):.4f}, Domain Loss {total_domain_loss / len(train_loader):.4f}, Train_acc {train_acc:.4f}, Test_acc {test_acc:.4f}")
                    wandb.log({'loss': loss, 'epoch': epoch, 'Class Loss': total_class_loss / len(train_loader), 'Domain Loss': total_domain_loss / len(train_loader), 'Train_acc': train_acc, 'Test_acc': test_acc})
            # scheduler.step()
        validation_accs[idx] = best_acc
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")
        wandb.log({'Fold':idx, 'best acc':validation_accs[idx]})
        run.finish()
    run = wandb.init(project='DANN_4_18_2',
                     name=f'Final',
                     config=args,
                     reinit=True)
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")
    wandb.log({'Average acc':acc_mean, 'acc std':acc_std})
    run.finish()


def train_DANN_resnet(args, train_dataset_all, test_datastet_all):
    validation_accs = np.zeros(15)
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    for idx in range(15):
        # run = wandb.init(project='DANN_resnet',
        #            name = f'fold:{idx}',
        #            config=args,
        #            reinit=True)

        train_dataset = train_dataset_all[idx]
        test_dataset = test_datastet_all[idx]

        train_dataset.data, test_dataset.data = normalize(train_dataset.data, test_dataset.data)

        # print(train_dataset.data.shape)
        # print(train_dataset.data[0])

        test_x, test_y = test_dataset.data.to(device), test_dataset.label.to(device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

        model = DANN_resnet(310, args.hidden_dim, 4, 2, args.lamda, momentum=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.3, last_epoch=-1)

        train_iter = iter(train_loader)
        best_acc = 0.
        for epoch in range(args.num_epoch):
            model.train()

            len_dataloader = min(len(train_loader), len(test_loader))
            total_class_loss = 0.
            total_domain_loss = 0.
            for i, data in enumerate(test_loader):
                p = float(i + epoch * len_dataloader) / args.num_epoch / len_dataloader
                lamda = 0.5 * (2. / (1. + np.exp(-10 * p)) - 1)

                try:
                    input_s, class_labels = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    input_s, class_labels = next(train_iter)
                input_s = input_s.unsqueeze(1)
                input_s = reshape_input(input_s)
                # print(input_s.shape)
                input_s, class_labels = input_s.to(device), class_labels.to(device)
                domain_source_labels = torch.zeros(len(input_s)).long().to(device)
                train_data = input_s.to(device).float(), class_labels.to(device), domain_source_labels.to(device)

                input_t, _ = data
                input_t = input_t.unsqueeze(1)
                input_t = reshape_input(input_t)

                input_t = input_t.to(device)
                domain_target_labels = torch.ones(len(input_t)).long().to(device)
                test_data = input_t.to(device).float(), domain_target_labels.to(device)

                optimizer.zero_grad()

                # pred_class_label, pred_domain_label = model.forward(input_s, lamda)
                # class_loss = criterion(pred_class_label, class_labels)
                # domain_source_loss = criterion(pred_domain_label, domain_source_labels)
                #
                # _, pred_domain_label = model.forward(input_t, lamda)
                # domain_target_loss = criterion(pred_domain_label, domain_target_labels)
                #
                # domain_loss = domain_source_loss + domain_target_loss
                class_loss, domain_loss = model.compute_loss(train_data, test_data, lamda)
                loss = class_loss + domain_loss

                # class_loss, domain_loss = model.compute_loss(train_data, test_data, lamda)
                # loss = class_loss + domain_loss

                loss.backward()
                optimizer.step()


                total_class_loss += class_loss.detach().cpu().numpy()
                total_domain_loss += domain_loss.detach().cpu().numpy()

            if epoch % args.display_epoch == 0:
                with torch.no_grad():
                    model.eval()

                    train_acc = test_resnet(model=model, test_dataloader=train_loader, device=device, alpha=lamda)
                    test_acc = test_resnet(model=model, test_dataloader=test_loader, device=device, alpha=lamda)

                    # pred_class_label, _ = model(test_x.float(), lamda)
                    # _, test_y_pred = torch.max(pred_class_label, dim=1)
                    # test_acc = (test_y_pred == test_y).sum().item() / len(test_dataset)
                    if test_acc > best_acc:
                        best_acc = test_acc
                        filename = f"{args.model}_checkpoint.pt"
                        torch.save(model.state_dict(), os.path.join(checkpoint_dir, filename))
                    print(f"Epoch {epoch}, Class Loss {total_class_loss / len(train_loader):.4f}, Domain Loss {total_domain_loss / len(train_loader):.4f}, Train_acc {train_acc:.4f}, Test_acc {test_acc:.4f}")
                    # wandb.log({'loss': loss, 'epoch': epoch, 'Class Loss': total_class_loss / len(train_loader), 'Domain Loss': total_domain_loss / len(train_loader), 'Train_acc': train_acc, 'Test_acc': test_acc})
            scheduler.step()
        validation_accs[idx] = best_acc
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")
        # wandb.log({'Fold':idx, 'best acc':validation_accs[idx]})
        # run.finish()
    # run = wandb.init(project='DANN_resnet',
    #                  name=f'Final',
    #                  config=args,
    #                  reinit=True)
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")
    # wandb.log({'Average acc':acc_mean, 'acc std':acc_std})
    # run.finish