# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/7/2


import numpy as np
import torch
import tqdm


# def predict_prob(model, data_loader, device):
#     model.eval()
#     y_pred = []
#     with torch.no_grad():
#         for X_batch, y_batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
#             X_batch, X_batch2, y_batch = X_batch.to(device), y_batch.to(device)
#             y_out = model(X_batch)
#             y_pred.extend(y_out.tolist())
#     return y_pred
#
#
# def train_tool(model, optimizer, data_loader, criterion, device, log_interval=0):
#     model.train()
#     total_loss = 0
#     tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
#     for i, (X_batch, y_batch) in enumerate(tk0):
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         y_out = model(X_batch)
#         loss = criterion(y_out, y_batch.float())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         if log_interval:
#             if (i + 1) % log_interval == 0:
#                 tk0.set_postfix(loss=total_loss / log_interval)
#                 total_loss = 0
#     return total_loss / len(data_loader)
#
# def test_tool(model, data_loader, criterion, device):
#     model.eval()
#     y_true, y_pred = [], []
#     total_loss = 0
#     with torch.no_grad():
#         for X_batch, y_batch in data_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             y_out = model(X_batch)
#             loss = criterion(y_out, y_batch.float())
#             y_true.extend(y_batch.tolist())
#             y_pred.extend(y_out.tolist())
#             total_loss += loss.item()
#     avg_loss = total_loss / len(data_loader)
#     return avg_loss



import numpy as np
import pandas as pd
from datetime import datetime
import random
import torch
import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from torch import nn
from torch.autograd import Function
from joblib import Parallel, delayed
import itertools


def get_threshold(y_true, cv_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
        cv_pred = np.asarray(cv_pred)
    best_score = 0
    for i in range(480, 550, 1):
        threshold = i / 1000
        pred = np.round(cv_pred - threshold + 0.5)
        score = f1_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    print('best_score', best_score)
    print('best_threshold', best_threshold)
    print('label len(1)', len(cv_pred[np.round(cv_pred - best_threshold + 0.5) == 1]))
    print('label len(0)', len(cv_pred[np.round(cv_pred - best_threshold + 0.5) == 0]))
    return best_threshold

def predict_prob(model, data_loader, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch1, X_batch2, y_batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            X_batch1, X_batch2, y_batch = X_batch1.to(device), X_batch2.to(device), y_batch.to(device)
            y_out = model(X_batch1, X_batch2)
            y_pred.extend(y_out.tolist())
    return y_pred


def train_tool(model, optimizer, data_loader, criterion, device, log_interval=0):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (X_batch1, X_batch2, y_batch) in enumerate(tk0):
        # print(X_batch1.shape)
        # print(X_batch2.shape)
        # print(y_batch)
        X_batch1, X_batch2, y_batch = X_batch1.to(device), X_batch2.to(device), y_batch.to(device)
        y_out = model(X_batch1, X_batch2)
        loss = criterion(y_out, y_batch.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if log_interval:
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
    return total_loss / len(data_loader)

def test_tool(model, data_loader, criterion, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    with torch.no_grad():
        for X_batch1, X_batch2, y_batch in data_loader:
            X_batch1, X_batch2, y_batch = X_batch1.to(device), X_batch2.to(device), y_batch.to(device)
            y_out = model(X_batch1, X_batch2)
            loss = criterion(y_out, y_batch.float())
            y_true.extend(y_batch.tolist())
            y_pred.extend(y_out.tolist())
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    best_threshold = get_threshold(y_true, y_pred)
    y_pred_label = np.where(y_pred >= best_threshold, 1.0, 0.0)
    f_score = f1_score(y_true, y_pred_label)

    return (best_threshold, f_score), avg_loss