import os
import numpy as np
import argparse
from os.path import join as pjoin
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Subset, Dataset, DataLoader, random_split

from data import TrainDataset
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Train Step",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--lags', type=int, default=5)
parser.add_argument('--stock_code', type=str)
parser.add_argument('--verbose', type=str2bool, default=False)
args = parser.parse_args()


def fgsm_attach(feature, feature_grad, epsilon=1e-3):
    # TODO: feature_grad.sign()
    # batch L2 normalization
    feature_grad = F.normalize(feature_grad, dim=0, p=2)
    perturbed_feature = feature + epsilon * feature_grad
    return perturbed_feature


def get_Acc(preds, trues):
    preds = [1. if pred > 0.5 else 0. for pred in preds]
    score = [p == t for p, t in zip(preds, trues)]
    return sum(score) / (len(score) + 1e-8)

def get_MCC(preds, trues):
    preds = [1. if pred > 0.5 else 0. for pred in preds]

    TP = sum([p*t for p, t in zip(preds, trues)])
    TN = sum([(1-p)*(1-t) for p, t in zip(preds, trues)])
    FP = sum([p*(1-t) for p, t in zip(preds, trues)])
    FN = sum([(1-p)*t for p, t in zip(preds, trues)])
    MCC = (TP*TN - FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return MCC


class Train:

    def __init__(self, args, dataset=None, model_cls=None):
        self.epochs = args.epochs
        self.verbose = args.verbose
        self.epsilon = 0.001

        if dataset is None:
            dataset = TrainDataset(stock_code=args.stock_code,
                                   lags=args.lags)

        data_len = len(dataset)
        self.train_num = int(data_len * 0.85)
        self.vali_num = data_len - self.train_num
        trainset, valiset = random_split(dataset, [self.train_num,
                                                   self.vali_num])

        self.train_loader = DataLoader(trainset, batch_size=args.batch_size,
                                       shuffle=True, num_workers=2)
        self.vali_loader = DataLoader(valiset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=2)

        self.batch_size = args.batch_size
        if model_cls is None:
            self.model = Model()
        else:
            self.model = model_cls() # for hyperoptimizer

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.model_path = pjoin("weight", args.stock_code + ".pt")

    def train(self, epoch, criterion):
        self.model.train()
        total_loss = 0.0
        for batch, (X, y) in enumerate(self.train_loader):
            X = X.to(device)
            y = y.to(device)
            score, feature = self.model(X)
            loss = criterion(score, y)

            self.optimizer.zero_grad()
            feature_grad = torch.autograd.grad(loss, [feature],
                                               retain_graph=True)[0]

            perturbed_feature = fgsm_attach(feature, feature_grad,
                                            epsilon=self.epsilon)
            adv_score = self.model.perturbed_forward(perturbed_feature)
            adv_loss = criterion(adv_score, y)
            loss = loss + adv_loss
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item() * X.size(0)
        total_loss /= self.train_num
        return total_loss

    def validate(self, epoch, loss_func):
        self.model.eval()
        total_loss = 0.0
        preds, trues = [], []
        for batch, (X, y) in enumerate(self.vali_loader):
            X = X.to(device)
            y = y.to(device)
            score, _ = self.model(X)
            trues += y.tolist()
            preds += score.tolist()
            loss = loss_func(score, y)
            total_loss += loss.item() * X.size(0)
        total_loss /= self.vali_num

        template = "LOSS: {:.4}, ACC: {:.4}, MCC: {:.4}"
        print(template.format(total_loss, get_Acc(preds, trues),
                              get_MCC(preds, trues)))
        return total_loss

    def run(self):
        loss_func = nn.BCELoss()
        if self.verbose:
            print(self.model)
        tot_vali_loss = np.inf
        for epoch in range(self.epochs):
            train_loss = self.train(epoch, loss_func)
            vali_loss = self.validate(epoch, loss_func)
            if self.verbose:
                print("train loss: {:.4} vali loss: {:.4}".format(train_loss, vali_loss))
            if tot_vali_loss > vali_loss:
                tot_vali_loss = vali_loss
                torch.save(self.model.state_dict(), self.model_path)


if __name__ == "__main__":
    train = Train(args)
    train.run()

