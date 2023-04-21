import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
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


def train_ADDA(args, train_dataset_all, test_dataset_all):
    LEARNING_RATE = args.lr
    NUM_EPOCH_PRE = args.pretrain_epoch
    NUM_EPOCH = args.num_epoch
    CUDA = True
    BATCH_SIZE = args.batch_size
    BETA1 = 0.5
    BETA2 = 0.9
    BATCHNORM_TRACK = False
    MOMENTUM = 0.5
    # TEST_IDX = 14

    pre_acc_list = np.zeros(15)
    da_acc_list = np.zeros(15)
    for TEST_IDX in range(15):
        run = wandb.init(project='ADDA_4_20',
                         name=f'fold:{TEST_IDX}',
                         config=args,
                         reinit=True)
        print(f'###### test idx {TEST_IDX} ######')

        # train source feature extractor and classifier on source domain data
        source_feature_extractor = FeatureExtractor(track_running_stats=BATCHNORM_TRACK, momentum=MOMENTUM)
        classifier = Classifier(track_running_stats=BATCHNORM_TRACK, momentum=MOMENTUM)
        criterion = nn.CrossEntropyLoss()

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
            train_acc = test_adda(source_feature_extractor, classifier, dataloader_source)
            acc = round((pred_class == t_label).float().mean().cpu().numpy().tolist(), 4)

            print(f'Pretrain Epoch: {epoch}, Train ACC: {train_acc:.4f}, Test Accuracy: {acc:.4f}')
            wandb.log({'Pretrain Epoch': epoch, 'Pretrain Train ACC': train_acc, 'Pretrain Test Accuracy': acc, 'Pretrain loss':loss})
        pre_acc_list[TEST_IDX] = acc

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

        dataloader_source = DataLoader(dataset=dataset_source, batch_size=BATCH_SIZE, shuffle=True, num_workers=3, drop_last=True)
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=BATCH_SIZE, shuffle=True, num_workers=3, drop_last=True)

        # accuracies = []
        best_acc = 0.0
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

                with torch.no_grad():
                    train_acc = test_adda(source_feature_extractor, classifier, dataloader_source)
                    acc = test_adda(target_feature_extractor, classifier, dataloader_target)
                if acc > best_acc:
                    best_acc = acc

                optimizer_disc.zero_grad()
                feat_src = source_feature_extractor(s_input)
                feat_tgt = target_feature_extractor(t_input)
                feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

                pred_src = discriminator(feat_src)
                pred_tgt = discriminator(feat_tgt)

                # pred_concat = discriminator(feat_concat.detach())

                # alpha = torch.rand(BATCH_SIZE, 1).to(device)
                # alpha = alpha.expand(feat_src.size())
                # interpolates = alpha * feat_src + (1 - alpha) * feat_tgt
                # interpolates = autograd.Variable(interpolates, requires_grad=True)
                #
                # disc_interpolates = discriminator(interpolates)
                # gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                #                           grad_outputs=torch.ones(disc_interpolates.size()).to(interpolates.device),
                #                           create_graph=True, retain_graph=True, only_inputs=True)[0]
                # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()


                label_src = torch.zeros(feat_src.shape[0]).long().to(device)
                label_tgt = torch.ones(feat_tgt.shape[0]).long().to(device)
                # label_concat = torch.cat((label_src, label_tgt), dim=0)

                # if CUDA:
                #     label_concat = label_concat.cuda()
                # loss_disc = criterion(pred_concat, label_concat)
                loss_disc = pred_tgt.mean() - pred_src.mean()
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

                loss_tgt = -criterion(pred_tgt, label_tgt)
                loss_tgt.backward()
                optimizer_tfe.step()


            # discriminator.eval()
            # target_feature_extractor.eval()
            # t_input, t_label = dataset_target[:]
            # if CUDA:
            #     t_input = t_input.cuda()
            #     t_label = t_label.cuda()
            #
            # with torch.no_grad():
            #     pred_class_score = classifier(target_feature_extractor(t_input))

            # pred_class = pred_class_score.max(1)[1]
            #
            # acc = round((pred_class == t_label).float().mean().cpu().numpy().tolist(), 4)
            # accuracies.append(acc)
            print(f'Train Epoch: {epoch}, Train acc: {train_acc:.4f}, Accuracy: {acc:.4f}')
            wandb.log({'Train Epoch': epoch, 'Train ACC': train_acc, 'Test Accuracy': acc,
                       'loss_tgt': loss_tgt, 'loss_disc': loss_disc})

        da_acc_list[TEST_IDX] = best_acc
        wandb.log({'best acc':best_acc})
        run.finish()
    run = wandb.init(project='ADDA_4_20',
                     name=f'Final',
                     config=args,
                     reinit=True)
    # print(f'Pre Accuracy List:\n{pre_acc_list}')
    print(f'Pre Mean Accuracy: {np.mean(pre_acc_list):.4f}, Pre std: {np.std(pre_acc_list):.4f}')

    # print(f'DA Accuracy list:\n{da_acc_list}')
    print(f'DA Mean accuracy: {np.mean(da_acc_list):.4f}, DA std: {np.std(da_acc_list):.4f}')
    wandb.log({'pre acc': np.mean(pre_acc_list), 'pre std': np.std(pre_acc_list), 'da acc': np.mean(da_acc_list), 'da std': np.std(da_acc_list)})
    run.finish()
    # plt.plot(range(1, NUM_EPOCH+1), accuracies)
    # plt.savefig(f'ADDA/results/{BETA1}_{BETA2}_test0_{pretrain_acc:.3f}.png')