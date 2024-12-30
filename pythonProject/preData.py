import pandas as pd
import tqdm


# 定义去除不合理值的函数
def remove_outliers(data, method="iqr", threshold=1.5):
    """
    去除不合理值（异常值）
    参数:
    - data: DataFrame 数据
    - method: 异常值检测方法（目前支持 'iqr'）
    - threshold: 阈值系数，默认 1.5 对应 IQR 方法
    返回:
    - 清洗后的数据
    """
    # for column in data.select_dtypes(include=[np.number]).columns:
    #     if column == "id":  # 跳过 id 列
    #         continue
    #     if method == "iqr":
    #         # 计算四分位数
    #         Q1 = data[column].quantile(0.25)
    #         Q3 = data[column].quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - threshold * IQR
    #         upper_bound = Q3 + threshold * IQR
    #         # 筛选数据范围
    #         data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    # return data
    column = 'PRICE VAR [%]'
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    # 筛选数据范围
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data


# 读取两个 CSV 文件
file_path1 = "dataSet/train.csv"  # 替换为第一个 CSV 文件路径
file_path2 = "dataSet/test.csv"  # 替换为第二个 CSV 文件路径
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# 假设需要编号的列名相同，例如 'column_name'
column_to_encode = 'Sector'  # 替换为你的目标列名

# 合并两列，进行统一编码
all_strings = pd.concat([df1[column_to_encode], df2[column_to_encode]])
all_encoded, unique_strings = pd.factorize(all_strings)

# 分离编码，分配回两个数据框
df1['Sector'] = all_encoded[:len(df1)]
df2['Sector'] = all_encoded[len(df1):]

# df1.loc[:,df1.isnull().all()] = 0
# df2.loc[:,df2.isnull().all()] = 0

df1.fillna(df1.mean(), inplace=True)
df2.fillna(df2.mean(), inplace=True)
# print(df1.shape)
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=20, weights='distance', metric='nan_euclidean', copy=True)
#
# df1_clean = imputer.fit_transform(df1)
# df1_clean = pd.DataFrame(df1_clean)
# df1_clean.columns = list(df1)
# df1 = df1_clean
#
# df2_clean = imputer.fit_transform(df2)
# df2_clean = pd.DataFrame(df2_clean)
# df2_clean.columns = list(df2)
# df2 = df2_clean

# 去除不合理数据
# df1 = remove_outliers(df1)

# 保存结果到新的 CSV 文件
output_path1 = "dataSet/output_train.csv"  # 替换为第一个输出文件路径
output_path2 = "dataSet/output_test.csv"  # 替换为第二个输出文件路径
df1.to_csv(output_path1, index=False)
df2.to_csv(output_path2, index=False)

print(f"编码完成，结果已保存到 {output_path1} 和 {output_path2}")
