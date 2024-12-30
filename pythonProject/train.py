import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import csv
import os

from sklearn.preprocessing import OneHotEncoder

myseed = 3900
batch_size = 500
n_epochs = 10000
early_stop = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

cur_path = "D:\桌面\大三上\机器学习\个人大作业\\22371461_钟芳梽_个人作业_股票预测\pythonProject"

class MyDataset(Dataset):
    def __init__(self,
            path,
            mode='train'):
        self.mode = mode

        with open(path, 'r') as fp:
            data = list(csv.reader(fp))

        data = np.array(data[1:])[:, 1:]
        categories = data[:, -2].reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = encoder.fit_transform(categories)
        data = np.hstack([data[:, :-2], one_hot_encoded, data[:, -1:]])

        if mode == "test":
            data = data[:, :-1]
        data[data == ""] = np.nan
        data = data.astype(float)

        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_mean, inds[1])

        if mode == "test":
            self.data = self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, :-1]

            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 100 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 100 == 0]

            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        self.data[:, :] = \
            (self.data[:, :] - self.data[:, :].mean(dim=0, keepdim=True)) \
            / (self.data[:, :].std(dim=0, keepdim=True) + 1e-8)

        self.dim = self.data.shape[1]

        print(f'load {mode} dataset with dim {self.dim}, size {self.data.__len__()}')


        with open(os.path.join(cur_path, f"{mode}Data.csv"), 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]
        

def get_dataloader(path, mode):
    dataset = MyDataset(os.path.join(cur_path,"dataSet",path), mode)
    return DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        pin_memory=True)  

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(500, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1000, 2000),
            nn.BatchNorm1d(2000),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(500, 1),
        )

        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)    


def dev(dv_set, model, device):
    model.eval()                              
    total_loss = 0
    for x, y in dv_set:                         
        x, y = x.to(device), y.to(device)
        with torch.no_grad():                  
            pred = model(x)                     
            mse_loss = model.cal_loss(pred, y)  
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)            

    return total_loss 

def test(tt_set, model, device):
    model.eval()                              
    preds = []
    for x in tt_set:                            
        x = x.to(device)                       
        with torch.no_grad():                  
            pred = model(x)                     
            preds.append(pred.detach().cpu())  
    preds = torch.cat(preds, dim=0).numpy()    
    return preds   

def train(tr_set, dv_set, model,device):
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.1)

    early_stop_cnt = 0
    epoch = 0
    min_mse = torch.inf
    while epoch < n_epochs:
        model.train()      
        train_mse = 0           
        for x, y in tr_set:                   
            optimizer.zero_grad()               
            x, y = x.to(device), y.to(device)   
            # print(x.size())
            pred = model(x)      
            # print(pred)             
            mse_loss = model.cal_loss(pred, y)
            train_mse += mse_loss
            mse_loss.backward()                
            optimizer.step()                 

        dev_mse = dev(dv_set, model, device)
        print(f"epoch{epoch} train_mse:{train_mse / batch_size} dev_mse:{dev_mse}")
        if dev_mse < min_mse:
            min_mse = dev_mse
            print(f'Saving model (epoch = {epoch + 1}, loss = {np.sqrt(min_mse)})')
            torch.save(model.state_dict(), os.path.join(cur_path, "model", "model.pth"))
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            torch.save(model.state_dict(), os.path.join(cur_path, "model", "model.pth"))

        epoch += 1
        if early_stop_cnt > early_stop:
            break

    print('Finished training after {} epochs'.format(epoch))

tr_set = get_dataloader("train.csv", 'train')
dv_set = get_dataloader("train.csv", 'dev')
tt_set = get_dataloader("test.csv", 'test')

model = NeuralNet(tr_set.dataset.dim).to(device)

train(tr_set, dv_set, model, device)

model.load_state_dict(torch.load(os.path.join(cur_path, "model", "model.pth")))

preds = []
for x in tt_set:
    x = x.to(device)
    with torch.no_grad():
        pred = model(x)
        preds.append(pred.detach().cpu())
preds = torch.cat(preds, dim=0).numpy()

with open(os.path.join(cur_path, "submission.csv"), "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(['id', 'PRICE VAR [%]'])
    for i,pred in enumerate(preds):
        writer.writerow([i+1, pred])