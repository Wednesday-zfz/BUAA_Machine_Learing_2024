import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zipfile
import torch.optim as optim
import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_file_path = "dataSet/output_train.csv"
test_file_path = "dataSet/output_test.csv"

myseed = 3900
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

class DataSet:
    def __init__(self, train_path, test_path):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        #print(train_data.shape)

        X = train_data.iloc[:, 1:-1].values
        y = train_data.iloc[:, -1].values
        X_test = test_data.iloc[:, 1:-1].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)


        X_train = X[[i for i in range(len(X)) if i % 100 != 0]]
        y_train = y[[i for i in range(len(X)) if i % 100!= 0]]
        X_val = X[[i for i in range(len(X)) if i % 100 == 0]]
        y_val = y[[i for i in range(len(X)) if i % 100 == 0]]
        #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=966)

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 转换为二维张量
        self.y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)

    def get_data(self, type):
        if type == "train":
            return self.X_train, self.y_train
        elif type == "val":
            return self.X_val, self.y_val
        elif type == "test":
            return self.X_test


# # 读取训练数据
# train_data = pd.read_csv(train_file_path)
# test_data = pd.read_csv(test_file_path)
#
# X_train = train_data.iloc[:, 1:-1].values
# y_train = train_data.iloc[:, -1].values
# X_test = test_data.iloc[:, 1:-1].values
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# # 定义PCA对象，例如设置保留95%的方差
# pca = PCA(n_components=0.95)
# # 在训练数据上拟合PCA并进行降维
# X_train = pca.fit_transform(X_train)
# # 在测试数据上应用相同的PCA变换
# X_test = pca.transform(X_test)
#
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 转换为二维张量
# X_test = torch.tensor(X_test, dtype=torch.float32)
#
# dataset = torch.utils.data.TensorDataset(X_train, y_train)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)


class FullyConnectedNN(nn.Module):
    def __init__(self, input_size):
        super(FullyConnectedNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return x





def package(predictions, is_cuda=False):
    # 保存预测结果到 CSV 文件
    output_file_path = "dataSet/submission.csv"
    output_pd = pd.read_csv(output_file_path)
    if is_cuda:
        output_pd['PRICE VAR [%]'] = predictions.cpu().numpy()
    else:
        output_pd['PRICE VAR [%]'] = predictions
    # test_result = pd.read_csv(filepath_or_buffer='dataSet/test_result.csv')
    # test_result.fillna(0, inplace=True)
    # for index, row in test_result.iterrows():
    #     if row['PRICE VAR [%]'] != 0:
    #         output_pd.loc[index, 'PRICE VAR [%]'] = row['PRICE VAR [%]']

    output_pd.to_csv(output_file_path, index=False)

    # 创建一个 ZIP 文件并将 CSV 文件添加进去
    zip_file_path = "dataSet/submission.zip"
    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file_path, os.path.basename(output_file_path))  # 将 CSV 文件添加到 ZIP 文件中
        print(f"CSV 文件已成功添加到 ZIP 文件：{zip_file_path}")


def FullyConnectedNN_train(train=True):
    dataSet = DataSet(train_file_path, test_file_path)

    X_train, y_train = dataSet.get_data("train")
    X_test = dataSet.get_data("test")
    X_val, y_val = dataSet.get_data("val")

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=True)

    model = FullyConnectedNN(X_train.shape[1]).float().to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), weight_decay=0.1)

    if train:
        # 训练模型
        epochs = 10000
        min_rmse = torch.inf
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                # print(batch_X.shape, batch_y.shape)
                optimizer.zero_grad()
                outputs = model.forward(batch_X.to(device))
                # print(outputs)
                loss = criterion(outputs, batch_y.to(device))
                # print(loss)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
            model.eval()
            with torch.no_grad():
                predictions = model(X_val.to(device))
                #print(predictions.shape)
                #print(y_val.shape)
                # 计算 RMSE
                mse = mean_squared_error(y_val.cpu().numpy(), predictions.cpu().numpy())
                rmse = np.sqrt(mse)

                if epoch>100 :
                    print(f'epoch:{epoch+1}  rmse: {rmse}  min_rmse:{min_rmse}')
                    if rmse < min_rmse:
                        min_rmse = rmse
                        early_stop = 0
                        print(f'Saving model (epoch = {epoch + 1}, loss = {min_rmse})')
                        torch.save(model.state_dict(), "./model.ckpt")
                    else:
                        early_stop += 1
                    if early_stop > 1000:
                        torch.save(model.state_dict(), "./model_final.ckpt")
                        break
    model.load_state_dict(torch.load("./model.ckpt"))
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device))

    package(predictions, is_cuda=True)


if __name__ == '__main__':
    # model = FullyConnectedNN_train()
    # x = input("1:随机森林 2:神经网络\n")
    # if x == "1":
    #     RandForest_train()
    #     exit()
    # elif x == "2":
    FullyConnectedNN_train(True)
    exit()








def RandForest_train():
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    X_test = test_data.iloc[:, 1:-1].values
    rf = RandForest(n_estimators=10, max_depth=10, random_state=42, train_data=train_data)
    rf.train()
    rf.evaluate()
    rf.predict(X_test)



class RandForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=400, train_data=None):
        """初始化随机森林分类器"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.imputer = SimpleImputer(strategy='mean')

        shuffled_train_data = train_data.sample(frac=1, random_state=420)
        X = shuffled_train_data.iloc[:, 1:-2].values
        y = shuffled_train_data.iloc[:, -1].values

        cnt = int(len(X) * 0.99)
        self.X_train, self.y_train = X[:cnt], y[:cnt]
        self.X_test, self.y_test = X[cnt:], y[cnt:]

        # self.X_train = self.imputer.transform(self.X_train)
        # self.X_test = self.imputer.transform(self.X_test)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse:.4f}")

    def predict(self, X_result):
        predictions = self.model.predict(X_result)
        package(predictions)
        return predictions