from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pickle as pkl
from copy import deepcopy
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from utils import load_data, accuracy, dense_tensor_to_sparse, resample, makedirs
from utils_inductive import transform_dataset_by_idx
from models import HGAT
import os, gc, sys
from print_log import Logger

logdir = "log/"
savedir = 'model/'
embdir = 'embeddings/'
makedirs([logdir, savedir, embdir])

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

write_embeddings = True
HOP = 2

dataset = 'example'

LR = 0.01 if dataset == 'snippets' else 0.005
DP = 0.95 if dataset in ['agnews', 'tagmynews'] else 0.8
WD = 0 if dataset == 'snippets' else 5e-6
LR = 0.05 if 'multi' in dataset else LR
DP = 0.5 if 'multi' in dataset else DP
WD = 0 if 'multi' in dataset else WD

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=LR,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=WD,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=DP,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--inductive', type=bool, default=False,
                    help='Whether use the transductive mode or inductive mode. ')
parser.add_argument('--dataset', type=str, default=dataset,
                    help='Dataset')
parser.add_argument('--repeat', type=int, default=1,
                    help='Number of repeated trials')
parser.add_argument('--node', action='store_false', default=True,
                    help='Use node-level attention or not. ')
parser.add_argument('--type', action='store_false', default=True,
                    help='Use type-level attention or not. ')

args = parser.parse_args()
dataset = args.dataset

args.cuda = not args.no_cuda and torch.cuda.is_available()
sys.stdout = Logger(logdir + "{}.log".format(dataset))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

loss_list = dict()

def margin_loss(preds, y, weighted_sample=False):
    nclass = y.shape[1]
    preds = preds[:, :nclass]
    y = y.float()
    lam = 0.25
    m = nn.Threshold(0., 0.)
    loss = y * m(0.9 - preds) ** 2 + \
        lam * (1.0 - y) * (m(preds - 0.1) ** 2)

    if weighted_sample:
        n, N = y.sum(dim=0, keepdim=True), y.shape[0]
        weight = torch.where(y == 1, n, torch.zeros_like(loss))
        weight = torch.where(y != 1, N-n, weight)
        weight = N / weight / 2
        loss = torch.mul(loss, weight)

    loss = torch.mean(torch.sum(loss, dim=1))
    return loss

def nll_loss(preds, y):
    y = y.max(1)[1].type_as(labels)
    return F.nll_loss(preds, y)

def evaluate(preds_list, y_list):
    nclass = y_list.shape[1]
    preds_list = preds_list[:, :nclass]
    if not preds_list.device == 'cpu':
        preds_list, y_list = preds_list.cpu(), y_list.cpu()

    threshold = 0.5
    multi_label = 'multi' in dataset
    if multi_label:
        y_list = y_list.numpy()
        preds_probs = preds_list.detach().numpy()
        preds = deepcopy(preds_probs)
        preds[np.arange(preds.shape[0]), preds.argmax(1)] = 1.0
        preds[np.where(preds >= threshold)] = 1.0
        preds[np.where(preds < threshold)] = 0.0
        [precision, recall, F1, support] = \
            precision_recall_fscore_support(y_list[preds.sum(axis=1) != 0], preds[preds.sum(axis=1) != 0],
                                            average='micro')
        [precision_ma, recall_ma, F1_ma, support] = \
            precision_recall_fscore_support(y_list[preds.sum(axis=1) != 0], preds[preds.sum(axis=1) != 0],
                                            average='macro')
        ER = accuracy_score(y_list, preds) * 100

        report = classification_report(y_list, preds, digits=5)

        print(' ER: %6.2f' % ER,
              'mi-ma: P: %5.1f %5.1f' % (precision*100, precision_ma*100),
              'R: %5.1f %5.1f' % (recall*100, recall_ma*100),
              'F1: %5.1f %5.1f' % (F1*100, F1_ma*100),
              end="")
        return ER, report
    else:
        y_list = y_list.numpy()
        preds_probs = preds_list.detach().numpy()
        preds = deepcopy(preds_probs)
        preds[np.arange(preds.shape[0]), preds.argmax(1)] = 1.0
        preds[np.where(preds < 1)] = 0.0
        [precision, recall, F1, support] = \
            precision_recall_fscore_support(y_list, preds, average='macro')
        ER = accuracy_score(y_list, preds) * 100
        print(' Ac: %6.2f' % ER,
              'P: %5.1f' % (precision*100),
              'R: %5.1f' % (recall*100),
              'F1: %5.1f' % (F1*100),
              end="")
        return ER, F1

LOSS = margin_loss if 'multi' in dataset else nll_loss


def train(epoch,
          input_adj_train, input_features_train, idx_out_train, idx_train,
          input_adj_val, input_features_val, idx_out_val, idx_val):
    print('Epoch: {:04d}'.format(epoch+1), end='')
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(input_features_train, input_adj_train)

    if isinstance(output, list):
        O, L = output[0][idx_out_train], labels[idx_train]
    else:
        O, L = output[idx_out_train], labels[idx_train]
    loss_train = LOSS(O, L)
    print(' | loss: {:.4f}'.format(loss_train.item()), end='')
    acc_train, f1_train = evaluate(O, L)
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(input_features_val, input_adj_val)
    if isinstance(output, list):
        loss_val = LOSS(output[0][idx_out_val], labels[idx_val])
        print(' | loss: {:.4f}'.format(loss_val.item()), end='')
        results = evaluate(output[0][idx_out_val], labels[idx_val])
    else:
        loss_val = LOSS(output[idx_out_val], labels[idx_val])
        print(' | loss: {:.4f}'.format(loss_val.item()), end='')
        results = evaluate(output[idx_out_val], labels[idx_val])
    print(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] = [loss_train.item()]

    if 'multi' in dataset:
        acc_val, res_line = results
        return float(acc_val.item()), res_line
    else:
        acc_val, f1_val = results
        return float(acc_val.item()), float(f1_val.item())


def test(epoch, input_adj_test, input_features_test, idx_out_test, idx_test):
    print(' '*90 if 'multi' in dataset else ' '*65, end='')
    t = time.time()
    model.eval()
    output = model(input_features_test, input_adj_test)

    if isinstance(output, list):
        loss_test = LOSS(output[0][idx_out_test], labels[idx_test])
        print(' | loss: {:.4f}'.format(loss_test.item()), end='')
        results = evaluate(output[0][idx_out_test], labels[idx_test])
    else:
        loss_test = LOSS(output[idx_out_test], labels[idx_test])
        print(' | loss: {:.4f}'.format(loss_test.item()), end='')
        results = evaluate(output[idx_out_test], labels[idx_test])
    print(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] += [loss_test.item()]

    if 'multi' in dataset:
        acc_test, res_line = results
        return float(acc_test.item()), res_line
    else:
        acc_test, f1_test = results
        return float(acc_test.item()), float(f1_test.item())




path = '../data/'+ dataset +'/'
adj, features, labels, idx_train_ori, idx_val_ori, idx_test_ori, idx_map = load_data(path = path, dataset = dataset)
N = len(adj)

# Transductive的数据集变换
if args.inductive:
    print("Transfer to be inductive.")

    # resample
    # 之前的数据集划分：  训练集20 * class   验证集1000   其他的测试集
    # 这里换成：         训练集不变，  验证集 -> 测试集，
    #                   原本的测试集拆出和训练集一样多的样本作为验证集，其余作为无标注样本
    idx_train,idx_unlabeled,idx_val,idx_test = resample(idx_train_ori,idx_val_ori,idx_test_ori,path,idx_map)

    # if experimentType == 'unlabeled':
    #     bias = int(idx_unlabeled.shape[0] * supPara)
    #     idx_unlabeled = idx_unlabeled[: bias]
    #     print("\n\tidx_train: {}, idx_unlabeled: {},\n\tidx_val: {}, idx_test: {}".format(
    #             idx_train.shape[0], idx_unlabeled.shape[0], idx_val.shape[0], idx_test.shape[0]))

    input_adj_train, input_features_train, idx_related_train, idx_out_train = \
            transform_dataset_by_idx(adj,features,torch.cat([idx_train, idx_unlabeled]),idx_train, hop=HOP)
    input_adj_val, input_features_val, idx_related_val, idx_out_val = \
            transform_dataset_by_idx(adj,features,idx_val,idx_val, hop=HOP)
    input_adj_test, input_features_test, idx_related_test, idx_out_test = \
            transform_dataset_by_idx(adj,features,idx_test,idx_test, hop=HOP)

    all_node_count = sum([_.shape[0] for _ in adj[0]])
    all_input_idx, all_related_idx = set(), set()
    for input_idx, related_idx in [ [torch.cat([idx_train, idx_unlabeled]), idx_related_train],
                                    [idx_val, idx_related_val],
                                    [idx_test, idx_related_test] ]:
        print("# input_nodes: {}, # related_nodes: {} / {}".format(
                        len(input_idx), len(related_idx), all_node_count))
        all_input_idx.update(input_idx.numpy().tolist())
        all_related_idx.update(related_idx.numpy().tolist())
    print("Sum: # input_nodes: {}, # related_nodes: {} / {}\n".format(
                        len(all_input_idx), len(all_related_idx), all_node_count))
else:
    print("Transfer to be transductive.")
    input_adj_train, input_features_train, idx_out_train = adj, features, idx_train_ori
    input_adj_val, input_features_val, idx_out_val = adj, features, idx_val_ori
    input_adj_test, input_features_test, idx_out_test = adj, features, idx_test_ori
    idx_train, idx_val, idx_test = idx_train_ori, idx_val_ori, idx_test_ori


if args.cuda:
    N = len(features)
    for i in range(N):
        if input_features_train[i] is not None:
            input_features_train[i] = input_features_train[i].cuda()
        if input_features_val[i] is not None:
            input_features_val[i] = input_features_val[i].cuda()
        if input_features_test[i] is not None:
            input_features_test[i] = input_features_test[i].cuda()
    for i in range(N):
        for j in range(N):
            if input_adj_train[i][j] is not None:
                input_adj_train[i][j] = input_adj_train[i][j].cuda()
            if input_adj_val[i][j] is not None:
                input_adj_val[i][j] = input_adj_val[i][j].cuda()
            if input_adj_test[i][j] is not None:
                input_adj_test[i][j] = input_adj_test[i][j].cuda()
    labels = labels.cuda()
    idx_train, idx_out_train = idx_train.cuda(), idx_out_train.cuda()
    idx_val, idx_out_val = idx_val.cuda(), idx_out_val.cuda()
    idx_test, idx_out_test = idx_test.cuda(), idx_out_test.cuda()



FINAL_RESULT = []
for i in range(args.repeat):
# Model and optimizer
    print("\n\nNo. {} test.\n".format(i+1))
    model = HGAT(nfeat_list=[i.shape[1] for i in features],
                 type_attention=args.type,
                 node_attention=args.node,
                    nhid=args.hidden,
                    nclass=labels.shape[1],
                    dropout=args.dropout,
                    gamma=0.1,
                    orphan=True,
                 )


    # print(model)
    print(len(list(model.parameters())))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()

    print(len(list(model.parameters())))
    print([i.size() for i in model.parameters()])
    # Train model
    t_total = time.time()
    vali_max = [0, [0, 0], -1]


    for epoch in range(args.epochs):
        vali_acc, vali_f1 = train(epoch,
                         input_adj_train, input_features_train, idx_out_train, idx_train,
                         input_adj_val, input_features_val, idx_out_val, idx_val)
        test_acc, test_f1 = test(epoch,
                        input_adj_test, input_features_test, idx_out_test, idx_test)
        if vali_acc > vali_max[0]:
            vali_max = [vali_acc, (test_acc, test_f1), epoch+1]
            with open(savedir + "{}.pkl".format(dataset), 'wb') as f:
                pkl.dump(model, f)

            if write_embeddings:
                makedirs([embdir])
                with open(embdir + "{}.emb".format(dataset), 'w') as f:
                    for i in model.emb.tolist():
                        f.write("{}\n".format(i))
                with open(embdir + "{}.emb2".format(dataset), 'w') as f:
                    for i in model.emb2.tolist():
                        f.write("{}\n".format(i))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if 'multi' in dataset:
        print("The best result is ACC: {0:.4f}, where epoch is {2}\n{1}\n".format(
            vali_max[1][0],
            vali_max[1][1],
            vali_max[2]))
    else:
        print("The best result is: ACC: {0:.4f} F1: {1:.4f}, where epoch is {2}\n\n".format(
            vali_max[1][0],
            vali_max[1][1],
            vali_max[2]))
    FINAL_RESULT.append(list(vali_max))

print("\n")
for i in range(len(FINAL_RESULT)):
    if 'multi' in dataset:
        print("{0}:\tvali:  {1:.5f}\ttest:  ACC: {2:.4f}, epoch={4}.\n{3}".format(
            i,
            FINAL_RESULT[i][0],
            FINAL_RESULT[i][1][0],
            FINAL_RESULT[i][1][1],
            FINAL_RESULT[i][2]))
    else:
        print("{}:\tvali:  {:.5f}\ttest:  ACC: {:.4f} F1: {:.4f}, epoch={}".format(
                        i,
            FINAL_RESULT[i][0],
            FINAL_RESULT[i][1][0],
            FINAL_RESULT[i][1][1],
            FINAL_RESULT[i][2]))