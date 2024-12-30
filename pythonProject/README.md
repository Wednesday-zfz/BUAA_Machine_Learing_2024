# README

### 安装依赖

```shell
pip install -r requirements.txt
```

### 运行方法

先运行preData.py，对数据集进行预处理

```
python preData.py
```

在运行NeuralNet.py，生成结果

```
python NeuralNet.py
```

原始数据、预处理数据以及结果都放在`\dataSet`目录下，其中结果会自动打包成`submission.zip`