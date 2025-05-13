import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class DataProcessing:
    def __init__(self):
        pass

    # 读取数据并分离自变量与目标变量
    def readData(self, file_train: str, file_test: str):
        df_train = pd.read_csv(file_train)
        df_test = pd.read_csv(file_test)
        
        # 分离自变量与目标变量
        X_train = df_train.drop(columns='y')
        y_train = df_train['y']

        X_test = df_test
    
        return X_train, y_train, X_test, df_train
        # return None

    # X44 和 X45 存在空白值，使用KNN进行插值，注意只能插入0或1
    def impute_KNN(self, X_train, X_test, n_neighbors=5):
        imputer = KNNImputer(n_neighbors=n_neighbors)

        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.fit_transform(X_test)

        # KNN 插值不一定为0或1（取的是平均值），手动调整为0或1
        for i in range(len(X_train_imputed)):
            X44 = X_train_imputed[i, -2]
            X45 = X_train_imputed[i, -1]
            if X44 <= 0.5:
                X_train_imputed[i, -2] = 0
            else:
                X_train_imputed[i, -2] = 1

            if X45 <= 0.5:
                X_train_imputed[i, -1] = 0
            else:
                X_train_imputed[i, -1] = 1

        for i in range(len(X_test_imputed)):
            X44 = X_test_imputed[i, -2]
            X45 = X_test_imputed[i, -1]
            if X44 <= 0.5:
                X_test_imputed[i, -2] = 0
            else:
                X_test_imputed[i, -2] = 1

            if X45 <= 0.5:
                X_test_imputed[i, -1] = 0
            else:
                X_test_imputed[i, -1] = 1

        return X_train_imputed, X_test_imputed

    # 对数值型变量进行标准化
    def standardization(self, X_train_imputed, X_test_imputed):
        scaler = StandardScaler()
        # 最后两列不用标准化
        X_train_to_scale = X_train_imputed[:, :-2]
        X_test_to_scale = X_test_imputed[:, :-2]
        X_train_scaled = scaler.fit_transform(X_train_to_scale)
        X_test_scaled = scaler.transform(X_test_to_scale)

        # 连接两部分
        X_train_imputed_scaled = np.hstack((X_train_scaled, X_train_imputed[:, -2:]))
        X_test_imputed_scaled = np.hstack((X_test_scaled, X_test_imputed[:, -2:]))
       
        return X_train_imputed_scaled, X_test_imputed_scaled
    

# 生成csv结果文件
def createSolution(X_test: pd.DataFrame, y_pred, name='solution'):
    # 创建一个包含id和y_pred的DataFrame
    solution_df = pd.DataFrame({'id': X_test.index, 'y': y_pred})
    
    # 将DataFrame导出为CSV文件
    solution_df.to_csv(f'{name}.csv', index=False)
    print(f"Solution file '{name}.csv' has been created.")
    return None

if __name__ == "main":
    print("wwwwwww")