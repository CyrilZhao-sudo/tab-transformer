# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/7/2


import numpy as np
import torch
import tqdm


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
    return avg_loss

