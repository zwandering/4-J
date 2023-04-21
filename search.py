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
from Dataset import SEEDIVDataset_indp, load_data_indp, reshape_input, load_data_indp_original, load_data_indp_original1
import copy
from models import ResNet18_, create_model, DANN, FeatureExtractor, Classifier, Discriminator, DANN_resnet
import wandb
# from models import create_model
import argparse
import math


def normalize(train_data, test_data):
    m = train_data.mean(dim=0, keepdim=True)
    s = train_data.std(dim=0, unbiased=True, keepdim=True)
    train_data = (train_data - m) / s

    m = test_data.mean(dim=0, keepdim=True)
    s = test_data.std(dim=0, unbiased=True, keepdim=True)
    test_data = (test_data - m) / s

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


def train():
    # train_dataset, test_dataset = load_data_indp('SEED-IV')
    train_dataset_all, test_dataset_all = load_data_indp_original1('SEED-IV')
    run = wandb.init()
    # config 不用固定值
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()

    train_dataset = train_dataset_all[12]
    test_dataset = test_dataset_all[12]
    train_dataset.data = train_dataset.data.reshape(train_dataset.data.shape[0], -1)
    test_dataset.data = test_dataset.data.reshape(test_dataset.data.shape[0], -1)

    train_dataset.data, test_dataset.data = normalize(train_dataset.data, test_dataset.data)
    # print(train_dataset.data.shape)
    # print(train_dataset.data[0])

    test_x, test_y = test_dataset.data.to(device), test_dataset.label.to(device)

    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

    model = DANN(310, 128, 4, 2, 0.5, momentum=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.3, last_epoch=-1)

    train_iter = iter(train_loader)
    best_acc = 0.
    for epoch in range(epochs):
        model.train()

        len_dataloader = min(len(train_loader), len(test_loader))
        total_class_loss = 0.
        total_domain_loss = 0.
        for i, data in enumerate(test_loader):
            p = float(i + epoch * len_dataloader) / epochs / len_dataloader
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

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()

                train_acc = test(model=model, test_dataloader=train_loader, device=device, alpha=lamda)
                test_acc = test(model=model, test_dataloader=test_loader, device=device, alpha=lamda)

                # pred_class_label, _ = model(test_x.float(), lamda)
                # _, test_y_pred = torch.max(pred_class_label, dim=1)
                # test_acc = (test_y_pred == test_y).sum().item() / len(test_dataset)
                if test_acc > best_acc:
                    best_acc = test_acc
                    # filename = f"{args.model}_checkpoint.pt"
                    # torch.save(model.state_dict(), os.path.join(checkpoint_dir, filename))
                print(
                    f"Epoch {epoch}, Class Loss {total_class_loss / len(train_loader):.4f}, Domain Loss {total_domain_loss / len(train_loader):.4f}, Train_acc {train_acc:.4f}, Test_acc {test_acc:.4f}")
                wandb.log({'loss': loss, 'epoch': epoch, 'Class Loss': total_class_loss / len(train_loader),
                           'Domain Loss': total_domain_loss / len(train_loader), 'Train_acc': train_acc,
                           'test_acc': test_acc})
        # scheduler.step()


def train_ADDA():
    train_dataset_all, test_dataset_all = load_data_indp_original1('SEED-IV')
    run = wandb.init()

    LEARNING_RATE = wandb.config.lr
    NUM_EPOCH_PRE = wandb.config.pretrain_epoch
    NUM_EPOCH = wandb.config.num_epoch
    CUDA = True
    BATCH_SIZE = wandb.config.batch_size
    BETA1 = wandb.config.beta1
    BETA2 = wandb.config.beta2
    BATCHNORM_TRACK = False
    MOMENTUM = 0.5
    # TEST_IDX = 14

    pre_acc_list = []
    da_acc_list = []
    for TEST_IDX in [12]:
        print(f'###### test idx {TEST_IDX} ######')

        # train source feature extractor and classifier on source domain data
        source_feature_extractor = FeatureExtractor(track_running_stats=BATCHNORM_TRACK, momentum=MOMENTUM)
        classifier = Classifier(track_running_stats=BATCHNORM_TRACK, momentum=MOMENTUM)
        criterion = nn.CrossEntropyLoss()
        # criterion  = nn.NLLLoss()

        if CUDA:
            source_feature_extractor = source_feature_extractor.cuda()
            classifier = classifier.cuda()

        optimizer = optim.Adam(list(source_feature_extractor.parameters()) + list(classifier.parameters()),
                               lr=LEARNING_RATE, betas=(BETA1, BETA2))

        dataset_source = train_dataset_all[TEST_IDX]
        dataset_target = test_dataset_all[TEST_IDX]

        dataset_source.data = dataset_source.data.reshape(dataset_source.data.shape[0], -1)
        dataset_target.data = dataset_target.data.reshape(dataset_target.data.shape[0], -1)

        dataset_source.data, dataset_target.data = normalize(dataset_source.data, dataset_target.data)

        dataloader_source = DataLoader(dataset=dataset_source, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

        # pretrain
        print('========== Pretrain Stage ==========')
        for epoch in range(NUM_EPOCH_PRE):
            source_feature_extractor.train()
            classifier.train()
            data_source_iter = iter(dataloader_source)
            for step, data_source in enumerate(dataloader_source):
                s_input, s_label = data_source

                optimizer.zero_grad()

                batch_size = len(data_source)

                if CUDA:
                    s_input = s_input.cuda()
                    s_label = s_label.cuda()

                pred = classifier(source_feature_extractor(s_input))
                loss = criterion(pred, s_label)

                loss.backward()
                optimizer.step()

            t_input, t_label = dataset_target[:]
            if CUDA:
                t_input = t_input.cuda()
                t_label = t_label.cuda()

            source_feature_extractor.eval()
            classifier.eval()
            with torch.no_grad():

                pred_class_score = classifier(source_feature_extractor(t_input))

            pred_class = pred_class_score.max(1)[1]

            acc = round((pred_class == t_label).float().mean().cpu().numpy().tolist(), 4)

            print(f'Pretrain Epoch: {epoch}, Accuracy: {acc:.4f}')
            wandb.log({'Pretrain Epoch': epoch, 'Pretrain Accuracy': acc, 'Pretrain Loss': loss})

        pre_acc_list.append(acc)

        # train target feature extractor and discriminator
        target_feature_extractor = FeatureExtractor(track_running_stats=BATCHNORM_TRACK, momentum=MOMENTUM)
        target_feature_extractor.load_state_dict(source_feature_extractor.state_dict())

        discriminator = Discriminator(track_running_stats=BATCHNORM_TRACK, momentum=MOMENTUM)

        if CUDA:
            target_feature_extractor.cuda()
            discriminator.cuda()

        optimizer_tfe = optim.Adam(target_feature_extractor.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizer_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

        dataset_source = train_dataset_all[TEST_IDX]
        dataset_target = test_dataset_all[TEST_IDX]

        dataset_source.data = dataset_source.data.reshape(dataset_source.data.shape[0], -1)
        dataset_target.data = dataset_target.data.reshape(dataset_target.data.shape[0], -1)

        dataset_source.data, dataset_target.data = normalize(dataset_source.data, dataset_target.data)

        dataloader_source = DataLoader(dataset=dataset_source, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

        # accuracies = []
        print('========== Train Stage ==========')
        for epoch in range(NUM_EPOCH):
            discriminator.train()
            target_feature_extractor.train()
            len_dataloader = min(len(dataloader_source), len(dataloader_target))
            data_source_iter = iter(dataloader_source)
            data_target_iter = iter(dataloader_target)

            for step in range(len_dataloader):
                try:
                    data_source = next(data_source_iter)
                except StopIteration:
                    data_source_iter = iter(dataloader_source)
                    data_source = next(data_source_iter)

                s_input, s_label = data_source
                try:
                    data_target = next(data_target_iter)
                except StopIteration:
                    data_target_iter = iter(dataloader_target)
                    data_target = next(data_target_iter)

                t_input, t_label = data_target

                if CUDA:
                    s_input = s_input.cuda()
                    t_input = t_input.cuda()

                optimizer_disc.zero_grad()
                feat_src = source_feature_extractor(s_input)
                feat_tgt = target_feature_extractor(t_input)
                feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

                pred_concat = discriminator(feat_concat.detach())

                label_src = torch.zeros(feat_src.shape[0]).long()
                label_tgt = torch.ones(feat_tgt.shape[0]).long()
                label_concat = torch.cat((label_src, label_tgt), dim=0)

                if CUDA:
                    label_concat = label_concat.cuda()
                loss_disc = criterion(pred_concat, label_concat)
                loss_disc.backward()

                optimizer_disc.step()

                # train target feature extractor
                optimizer_disc.zero_grad()
                optimizer_tfe.zero_grad()

                feat_tgt = target_feature_extractor(t_input)
                pred_tgt = discriminator(feat_tgt)
                label_tgt = torch.zeros(feat_tgt.shape[0]).long()

                if CUDA:
                    label_tgt = label_tgt.cuda()

                loss_tgt = criterion(pred_tgt, label_tgt)
                loss_tgt.backward()
                optimizer_tfe.step()

            discriminator.eval()
            target_feature_extractor.eval()
            t_input, t_label = dataset_target[:]
            if CUDA:
                t_input = t_input.cuda()
                t_label = t_label.cuda()

            with torch.no_grad():
                pred_class_score = classifier(target_feature_extractor(t_input))

            pred_class = pred_class_score.max(1)[1]

            acc2 = round((pred_class == t_label).float().mean().cpu().numpy().tolist(), 4)
            # accuracies.append(acc)
            print(f'Train Epoch: {epoch}, Accuracy: {acc2:.4f}')
            wandb.log({'Train Epoch': epoch, 'acc2': acc2, 'loss_tgt': loss_tgt, 'loss_disc': loss_disc})

        da_acc_list.append(acc2)

    print(f'Pre Accuracy List:\n{pre_acc_list}')
    print(f'Pre Mean Accuracy: {sum(pre_acc_list) / len(pre_acc_list):.4f}')
    wandb.log({'Pre Mean Accuracy': sum(pre_acc_list) / len(pre_acc_list)})

    print(f'DA Accuracy list:\n{da_acc_list}')
    print(f'DA Mean accuracy: {sum(da_acc_list) / len(da_acc_list):.4f}')
    wandb.log({'DA Mean accuracy': sum(da_acc_list) / len(da_acc_list)})

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = "./checkpoint"

seed = 42

if seed != None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


sweep_config = {
    'method': 'bayes',
    'metric': {
        'goal': 'maximize',
        'name': 'acc2'
    },
    'parameters': {
        'batch_size': {'values': [16, 32, 64, 128, 256, 512]},
        'pretrain_epoch': {'values': [50, 75, 100, 150, 200, 300]},
        'num_epoch': {'values': [50, 75, 100, 150, 200, 300]},
        'beta1':{'max': 1., 'min': 0.},
        'beta2':{'max': 1., 'min': 0.},
        'lr': {'max': 0.01, 'min': 0.00001}
    }
}

sweep_id = wandb.sweep(sweep_config, project="ADDA-sweep-crossloss")
# 初始化
wandb.agent(sweep_id, function=train_ADDA, count=100)
