from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.api as sm

class Models:
    def __init__(self, X_train, y_train, X_test, df_train):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.df_train = df_train
    
    def trainLR(self, summary='yes'):
        # 直接进行多元线性回归
        X_train = self.X_train
        model = sm.OLS(self.y_train, X_train).fit()

        # 给出summary
        if summary == 'yes':
            model.summary()

        X_test = self.X_test

        # 在训练集上预测
        y_pred_train = model.predict(X_train)

        # 在测试集上预测
        y_pred_test = model.predict(X_test)
        
        # 多元线性回归并不用于分类，需要将预测值转换为01变量
        for i in range(len(y_pred_train)):
            if y_pred_train[i] <= 0.5:
                y_pred_train[i] = 0
            else:
                y_pred_train[i] = 1

        for i in range(len(y_pred_test)):
            if y_pred_test[i] <= 0.5:
                y_pred_test[i] = 0
            else:
                y_pred_test[i] = 1
        
        # 在训练集上评估
        report = classification_report(self.y_train, y_pred_train)

        return y_pred_test, report

    # 多元逻辑回归
    def trainLgR(self, random_state=42):
        # 尝试直接进行多元逻辑回归，观察效果
        LR_model = LogisticRegression(random_state=random_state)

        LR_model.fit(self.X_train, self.y_train)

        # 评估拟合效果
        # 逻辑回归本质上是分类模型，predict 方法会返回离散值0或1
        y_train_pred = LR_model.predict(self.X_train)
        y_test_pred = LR_model.predict(self.X_test)

        report = classification_report(self.y_train, y_train_pred)

        return y_test_pred, report
        

if __name__ == "main":
    print("wwwwwww")