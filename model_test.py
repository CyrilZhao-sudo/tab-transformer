# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/7/5

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os, json
from datetime import datetime
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from src.utils import train_tool, test_tool, predict_prob
from src.model import TabTransformer

TAG = str(datetime.now())[:19]
DATA_TAG = '2021-07-02 21:08:19'
data = pd.read_hdf('/home/mi/PycharmProjects/mi-project/resources/kdxf/%s_data_mapped.hdf' %DATA_TAG, 'train')

train, test = train_test_split(data, train_size=0.8)

field_names = ['gender', 'age', 'province', 'city', 'model'] # make

with open('/home/mi/PycharmProjects/mi-project/resources/kdxf/%s_all_encode.json' %DATA_TAG, "r", encoding='utf8') as f:
    all_encode = json.load(f)
cat_field_dims = []
for filed in field_names:
    if filed in ['gender', 'age']:
        cat_field_dims.append(len(all_encode.get(filed)))
    else:
        cat_field_dims.append(len(all_encode.get(filed + '_encode')))

cons_field_name = ['app_size', 'ratio_age', 'ratio_province', 'ratio_city', 'ratio_model']

# normize
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
X_train_cons_norm= normalizer.fit_transform(train[cons_field_name].values)
train[cons_field_name] = X_train_cons_norm
X_test_cons_norm = normalizer.transform(test[cons_field_name].values)
test[cons_field_name] = X_test_cons_norm

print(test[cons_field_name].head())


train_dataset, test_dataset = TensorDataset(torch.from_numpy(train[field_names].values),
                                            torch.from_numpy(train[cons_field_name].values.astype(np.float32)),
                                            torch.from_numpy(train['label'].values)), \
                              TensorDataset(torch.from_numpy(test[field_names].values),
                                            torch.from_numpy(test[cons_field_name].values.astype(np.float32)),
                                            torch.from_numpy(test['label'].values))

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True), \
                                                      DataLoader(test_dataset, batch_size=1024, shuffle=False)


loss = torch.nn.BCELoss()

model = TabTransformer(cat_field_dims, len(cons_field_name), embed_dim=32, depth=2, n_heads=4, att_dropout=0.5, an_dropout=0.5, ffn_dropout=0.5, mlp_dims=[16, 16])

weight_parameters, bias_parameters = [], []
for name, param in model.named_parameters():
    if 'weight' in name:
        weight_parameters.append(param)
    if 'bias' in name:
        bias_parameters.append(param)
optimizer = Adam(params=[{'params': weight_parameters, 'weight_decay': 0.0003}, {'params': bias_parameters}], lr=0.005)

# tensorboard
PATH = './resources/'
writer = SummaryWriter(os.path.join(PATH, 'logs/%s_log' % TAG))
best_thresholds = []

for epoch in range(35):
    train_loss = train_tool(model, optimizer, train_dataloader, loss, device='cpu')
    test_score, test_loss = test_tool(model, test_dataloader, loss, device='cpu')
    print(f'\n Epoch {epoch + 1}  '
          f'train loss:{round(train_loss, 4)}. '
          f'test loss:{round(test_loss, 4)}')
    writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, global_step=epoch)
writer.close()


