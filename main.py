import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from Dataset import SEEDIVDataset_indp, load_data_indp, reshape_input, load_data_indp_norm, tmp
import copy
from models import ResNet18_, create_model, DANN

# from models import create_model
import argparse

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


def train_svm(args, train_dataset, test_dataset):
    validation_accs = np.zeros(15)
    for i in range(15):
        print(i)
        train_data = train_dataset[i]
        test_data = test_dataset[i]

        train_data.data = train_data.data.reshape(train_data.data.shape[0], -1)
        test_data.data = test_data.data.reshape(test_data.data.shape[0], -1)

        clf = make_pipeline(StandardScaler(), SVC(C=args.svm_c, kernel=args.svm_kernel, gamma='auto'))
        clf.fit(train_data.data, train_data.label)
        test_predict = clf.predict(test_data.data)
        # print(classification_report(np.squeeze(test_dataset.y), test_predict))
        validation_accs[i] = accuracy_score(test_data.label, test_predict)
        print(f"Fold {i} acc: {validation_accs[i]:.4f}")
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")
    os.makedirs('results', exist_ok=True)
    np.save('results/svm', validation_accs)


def train_resnet(args, train_dataset, test_dataset):
    num_epochs = args.num_epoch
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_acc = []
    best_models = []
    for i in range(15):
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
                for data, labels in test_dataloader:
                    data = data.unsqueeze(1)
                    data = reshape_input(data)
                    data, labels = data.to(device), labels.to(device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    # print(predicted, labels)
                    total += labels.size(0)
                    # print(predicted, labels)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                if accuracy > best_acc:
                    best_acc = accuracy
                    # Save the model with the highest accuracy
                    best_model = copy.deepcopy(model.state_dict())
                print(f"Subject {i + 1}, Epoch {epoch + 1}: Train Loss {train_loss:.6f}, Accuracy {accuracy:.2f}%")
        print(best_acc)
        total_acc.append(best_acc)
        best_models.append(best_model)
    acc = sum(total_acc) / len(total_acc)
    print(f"Total Accuracy {acc:.2f}%")

    os.makedirs(f"indp_acc_{acc:.2f}", exist_ok=True)
    for i in range(15):
        torch.save(best_models[i], f'indp_acc_{acc:.2f}/model_{i:02d}.pt')


def train_generalization(args, train_dataset_all, test_dataset_all):
    validation_accs = np.zeros(15)
    for idx in range(15):
        train_dataset = train_dataset_all[idx]
        test_dataset = test_dataset_all[idx]
        train_dataset.data = train_dataset.data.reshape(train_dataset.data.shape[0], -1)
        test_dataset.data = test_dataset.data.reshape(test_dataset.data.shape[0], -1)

        test_x, test_y = torch.tensor(test_dataset.data).to(device), torch.tensor(test_dataset.label).to(device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

        model = create_model(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_acc = 0.
        for epoch in range(args.num_epoch):
            model.train()
            total_loss = 0.
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                loss = model.compute_loss((x, y))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().cpu().numpy()
            if epoch % args.display_epoch == 0:
                model.eval()
                _, test_y_pred = torch.max(model(test_x.float()), dim=1)
                test_acc = (test_y_pred == test_y).sum().item() / len(test_dataset)
                if test_acc > best_acc:
                    best_acc = test_acc
                    filename = f"{args.model}_checkpoint.pt"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    file_path = os.path.join(checkpoint_dir, filename)
                    torch.save(model.state_dict(), file_path)
                print(f"Epoch {epoch}, Loss {total_loss / len(train_loader):.4f}, Acc {test_acc:.4f}")
        validation_accs[idx] = best_acc
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")

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

def train_DANN(args, train_dataset_all, test_datastet_all):
    validation_accs = np.zeros(15)
    criterion = nn.NLLLoss()
    for idx in range(15):
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

        test_iter = iter(test_loader)
        best_acc = 0.
        for epoch in range(args.num_epoch):
            model.train()
            total_class_loss = 0.
            total_domain_loss = 0.
            for i, data in enumerate(train_loader):
                p = float(i + epoch * len(test_loader)) / args.num_epoch / len(test_loader)
                lamda = 0.5 * (2. / (1. + np.exp(-10 * p)) - 1)

                input_s, class_labels = data
                input_s, class_labels = input_s.to(device), class_labels.to(device)
                domain_source_labels = torch.zeros(len(input_s)).long().to(device)
                try:
                    input_t, _ = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    input_t, _ = next(test_iter)

                domain_target_labels = torch.ones(len(input_t)).long().to(device)
                input_t = input_t.to(device)

                optimizer.zero_grad()

                pred_class_label, pred_domain_label = model.forward(input_s, lamda)
                class_loss = criterion(pred_class_label, class_labels)
                domain_source_loss = criterion(pred_domain_label, domain_source_labels)

                _, pred_domain_label = model.forward(input_t, lamda)
                domain_target_loss = criterion(pred_domain_label, domain_target_labels)

                domain_loss = domain_source_loss + domain_target_loss

                # class_loss, domain_loss = model.compute_loss(train_data, test_data, lamda)
                loss = class_loss + domain_source_loss + domain_target_loss
                
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
        validation_accs[idx] = best_acc
        print(f"Fold {idx} best acc: {validation_accs[idx]:.4f}")
    acc_mean = np.mean(validation_accs)
    acc_std = np.std(validation_accs)
    print(f"Average acc is: {acc_mean:.4f}±{acc_std:.4f}")


def train(args, train_dataset_all, test_datastet_all):
    validation_accs = np.zeros(15)
    criterion = nn.NLLLoss()
    for idx in range(15):
        train_dataset = train_dataset_all[idx]
        test_dataset = test_datastet_all[idx]
        train_dataset.data = train_dataset.data.reshape(train_dataset.data.shape[0], -1)
        test_dataset.data = test_dataset.data.reshape(test_dataset.data.shape[0], -1)

        train_dataset.data, test_dataset.data = normalize(train_dataset.data, test_dataset.data)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 256
        n_epoch = 100
        lr = 1e-3
        momentum = 0.5
        model = DANN(310, args.hidden_dim, 4, 2, args.lamda).to(device)
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=lr)
        class_loss = nn.NLLLoss()
        domain_loss = nn.NLLLoss()

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
        # print_dataset_info(train_dataloader, test_dataloader)
    
        train_acc_list = []
        test_acc_list = []
    
        for epoch in range(n_epoch):
            len_dataloader = min(len(train_dataloader), len(test_dataloader))
            source_iter = iter(train_dataloader)
            target_iter = iter(test_dataloader)
    
            model.train()
            for i in range(len_dataloader):
                p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
                alpha = 0.5 * (2. / (1. + np.exp(-10 * p)) - 1)
    
                sorce_data = source_iter._next_data()
                s_data, s_label = sorce_data
    
                optimizer.zero_grad()
                batch_size = len(s_label)
                s_domain_label = torch.zeros(batch_size).long()
    
                s_data = s_data.to(device)
                s_label = s_label.long().to(device)
                s_domain_label = s_domain_label.to(device)
    
                s_class_pred, s_domain_pred = model(input_data=s_data, lamda=alpha)
                s_class_loss = class_loss(s_class_pred, s_label)
                s_domain_loss = domain_loss(s_domain_pred, s_domain_label)
    
                target_data = target_iter._next_data()
                t_data, t_label = target_data
    
                batch_size = len(t_data)
                t_domain_label = torch.ones(batch_size).long()
    
                t_data = t_data.to(device)
                t_domain_label = t_domain_label.to(device)
    
                t_class_pred, t_domain_pred = model(input_data=t_data, lamda=alpha)
                t_domain_loss = domain_loss(t_domain_pred, t_domain_label)
    
                loss = s_class_loss + s_domain_loss + t_domain_loss
                # print("test_session_id: %d, iter_id: %d, s_class_loss: %.4f, s_domain_loss: %.4f, t_domain_loss: %.4f" % (id, i, s_class_loss, s_domain_loss, t_domain_loss))
                loss.backward()
                optimizer.step()
    
            train_acc = test(model=model, test_dataloader=train_dataloader, device=device, alpha=alpha)
            test_acc = test(model=model, test_dataloader=test_dataloader, device=device, alpha=alpha)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
    
            save_path = "model_" + str(id) + "_epoch_" + str(epoch) + ".pt"
            # torch.save(model, save_path)
    
            print("test_session_id: %d, epoch: %d, train_acc: %.4f, test_acc: %.4f" % (idx, epoch, train_acc, test_acc))
    
        max_train_acc = max(train_acc_list)
        max_test_acc = max(test_acc_list)
        max_train_acc_idx = train_acc_list.index(max_train_acc)
        max_test_acc_idx = test_acc_list.index(max_test_acc)
        print("test_session_id: %d, max_train_acc: %.4f @ epoch: %d, max_test_acc: %.4f @ epoch: %d" % (
        i, max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx))
    return max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="General Traning Pipeline")

    parser.add_argument("--model", type=str, default='SVM')
    parser.add_argument("--svm_c", type=float, default=.1)
    parser.add_argument("--svm_kernel", choices=['linear', 'rbf', 'sigmoid', 'poly'], default='linear')
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--triplet_weight", type=float, default=.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--display_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--penalty_weight", type=float, default=.5)
    parser.add_argument("--weight_decay", type=float, default=.0)
    parser.add_argument("--variance_weight", type=float, default=.5)
    parser.add_argument("--wgan_lamda", type=float, default=10.)
    parser.add_argument("--pretrain_epoch", type=int, default=500)
    parser.add_argument("--advtrain_iteration", type=int, default=10000)
    parser.add_argument("--critic_iters", type=int, default=10)
    parser.add_argument("--gen_iters", type=int, default=10000)
    parser.add_argument("--is_augmentation", action='store_true')
    parser.add_argument("--display_iters", type=int, default=10)

    args = parser.parse_args()

    # fix random seed for reproducibility
    # if args.seed != None:
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)
    #     np.random.seed(args.seed)
    #     torch.backends.cudnn.deterministic = True

    train_dataset, test_dataset = load_data_indp('SEED-IV')
    # train_dataset, test_dataset = tmp()

    print('Load Data Success!')

    if args.model == 'SVM':
        train_svm(args, train_dataset, test_dataset)
    elif args.model == 'ResNet':
        train_resnet(args, train_dataset, test_dataset)
    elif args.model == 'DANN' or args.model == 'SADA':
        train(args, train_dataset, test_dataset)
    # elif args.model == 'ADDA':
    #     train_adaptation(args, train_dataset, test_dataset)
    elif args.model == 'MLP' or args.model == 'IRM' or args.model == 'REx':
        train_generalization(args, train_dataset, test_dataset)
    else:
        raise ValueError("Unknown model type!")
