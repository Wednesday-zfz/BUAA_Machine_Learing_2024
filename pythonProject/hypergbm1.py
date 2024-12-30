import os
import pandas as pd
import numpy as np
import zipfile
from sklearn.preprocessing import StandardScaler
from hypergbm import make_experiment
from hypergbm.search_space import search_space_general
from hypernets.tabular.datasets import dsutils
from hypernets.core.callbacks import SummaryCallback

# 数据加载
train_file_path = "dataSet/output_train.csv"
test_file_path = "dataSet/output_test.csv"
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)
print(train_data.columns)

X_train = train_data.iloc[:, 1:]  # 去掉第一列
y_train = train_data.iloc[:, ]  # 目标列
X_test = test_data.iloc[:, 1:]  # 去掉第一列

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将标准化后的数据转换为 Pandas DataFrame，并恢复列名
X_train = pd.DataFrame(X_train, columns=train_data.columns[1:])  # 使用训练数据的列名
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test, columns=test_data.columns[1:])  # 使用测试数据的列名

# 使用 HyperGBM 进行自动调参
experiment = make_experiment(
    train_data=X_train,
    target='PRICE VAR [%]',  # 根据实际目标列名称设置目标
    search_space=search_space_general,  # 通用搜索空间
    task="regression",  # 回归任务
    eval_size=0.2,  # 验证集大小
    max_trials=50,  # 最大实验次数
    optimize_direction="min",  # 最小化损失
)

# 开始实验
estimator = experiment.run()

# 使用最佳模型进行预测
predictions = estimator.predict(X_test)

# 保存预测结果到 CSV 文件
output_file_path = "dataSet/submission.csv"
output_pd = pd.read_csv(output_file_path)
output_pd['PRICE VAR [%]'] = predictions
output_pd.to_csv(output_file_path, index=False)

# 创建 ZIP 文件
zip_file_path = "dataSet/submission.zip"
if os.path.exists(zip_file_path):
    os.remove(zip_file_path)
with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(output_file_path, os.path.basename(output_file_path))  # 将 CSV 文件添加到 ZIP 文件中
    print(f"CSV 文件已成功添加到 ZIP 文件：{zip_file_path}")
