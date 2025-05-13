import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class DataProcessing:
    def __init__(self):
        pass

    # 读取数据并分离自变量与目标变量
    def readData(self, file_train: str, file_test: str, show='yes'):
        df_train = pd.read_csv('train.csv')
        df_test = pd.read_csv('test.csv')

        if show == 'yes':
            df_train.head(5)
        
        # 分离自变量与目标变量
        X_train = df_train.drop(columns='y')
        y_train = df_train['y']

        X_test = df_test
    
        return X_train, y_train, X_test

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
                X44 = 0
            else:
                X44 = 1

            if X45 <= 0.5:
                X45 = 0
            else:
                X45 = 1

        for i in range(len(X_test_imputed)):
            X44 = X_test_imputed[i, -2]
            X45 = X_test_imputed[i, -1]
            if X44 <= 0.5:
                X44 = 0
            else:
                X44 = 1

            if X45 <= 0.5:
                X45 = 0
            else:
                X45 = 1

    # 对数值型变量进行标准化
    def standardization(self, X_train_imputed, X_test_imputed):
        scaler = StandardScaler()
        X_train_imputed_scaled = scaler.fit_transform(X_train_imputed)
        X_test_imputed_scaled = scaler.transform(X_test_imputed)
        return X_train_imputed_scaled, X_test_imputed_scaled
    

# 生成csv结果文件
def createSolution(X_test, y_pred, name='solution'):