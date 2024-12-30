import math
import os
import zipfile

import pandas as pd
from transformers.pipelines import values


def findAns():
    train = pd.read_csv(filepath_or_buffer='dataSet/train.csv', dtype={'id': int})
    test = pd.read_csv(filepath_or_buffer='dataSet/test.csv', dtype={'id': int})


    test = test[~(test.iloc[:,1:-2].isnull() | test.iloc[:,1:-2].eq(0)).all(axis=1)]
    train = train[~(train.iloc[:,1:-2].isnull() | train.iloc[:,1:-2].eq(0)).all(axis=1)]
    print(test.shape)
    result_pd = pd.merge(train, test, on=test.columns[1:-2].tolist(), suffixes=('_train', '_test'))
    result_pd.to_csv('dataSet/result_pd.csv')

    # result = []
    # for index1, row1 in test.iterrows():
    #     if math.isnan(row1[1]):
    #         continue
    #     for index2, row2 in train.iterrows():
    #         if row1[1] == row2[1]:
    #             result.append(row2)
    #             break
    #
    # result_pd = pd.DataFrame(data=result, columns=train.columns.tolist())
    # result_pd.to_csv('dataSet/result.csv')

def genTestResults():
    test = pd.read_csv(filepath_or_buffer='dataSet/test.csv', dtype={'id': int})
    result_pd = pd.read_csv(filepath_or_buffer='dataSet/result_pd.csv')
    # 根据id列合并两个DataFrame，只保留需要的列
    merged = pd.merge(test, result_pd[['id_test', 'PRICE VAR [%]_train']], left_on = 'id',right_on = 'id_test', how = 'left')

    # 将合并后的ans列的值赋给test的ans列
    test['PRICE VAR [%]'] = merged['PRICE VAR [%]_train']

    test.to_csv('dataSet/test_result.csv',index=False)



def package():
    # 保存预测结果到 CSV 文件

    test_result = pd.read_csv(filepath_or_buffer='dataSet/test_result.csv')
    test_result.fillna(0,inplace=True)
    output_file_path = "dataSet/submission.csv"
    output_pd = pd.read_csv(output_file_path)
    output_pd['PRICE VAR [%]'] = test_result['PRICE VAR [%]']
    output_pd.to_csv(output_file_path, index=False)

    # 创建一个 ZIP 文件并将 CSV 文件添加进去
    zip_file_path = "dataSet/submission.zip"
    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file_path, os.path.basename(output_file_path))  # 将 CSV 文件添加到 ZIP 文件中
        print(f"CSV 文件已成功添加到 ZIP 文件：{zip_file_path}")

if __name__ == '__main__':
    findAns()
