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

from data import StockDataset
from model import Model
from train import get_Acc, get_MCC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def save(path, content):
    with open(path, "w") as fout:
        fout.write(str(content)+"\n")

def test(args, dataset=None):

    if dataset is None:
        dataset = StockDataset(lags=args.lags,
                               is_regression=args.is_regression,
                               is_train=False)
    test_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    model = Model()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    loss_func = nn.MSELoss(reduction="mean") if args.is_regression else nn.BCELoss()

    total_loss = 0.0
    preds, trues = [], []
    for batch, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)
        score, _ = model(X)
        if not args.is_regression:
            score = torch.sigmoid(score)
        trues += y.tolist()
        preds += score.tolist()
        loss = loss_func(score, y)
        total_loss += loss.item() * X.size(0)
    total_loss /= len(dataset)

    if args.verbose and not args.is_regression:
        template = "LOSS: {:.4}, ACC: {:.4}, MCC: {:.4}"
        print(template.format(total_loss, get_Acc(preds, trues),
                              get_MCC(preds, trues)))
    if args.verbose and args.is_regression:
        print("LOSS: {:.4}".format(total_loss))
    if not args.is_regression:
        res = get_Acc(preds, trues)
        save(args.namespace, res)
        return res
    else:
        save(args.namespace, total_loss)
        return total_loss

def main():
    parser = argparse.ArgumentParser(description="Train Step",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_num', type=int, default=16)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--lags', type=int, default=5)
    parser.add_argument('--model_path', type=str, default="weight/model.pt")
    parser.add_argument('--namespace', type=str, default="res/model")
    parser.add_argument('--is_regression', type=str2bool, default=False)
    parser.add_argument('--verbose', type=str2bool, default=False)
    args = parser.parse_args()

    test(args)

if __name__ == "__main__":
    main()

