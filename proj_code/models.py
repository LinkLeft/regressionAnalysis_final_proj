from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

class Models:
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

    # 多元逻辑回归
    def LR(self, random_state=42):
        # 尝试直接进行多元逻辑回归，观察效果
        LR_model = LogisticRegression(random_state=random_state)

        LR_model.fit(self.X_train, self.y_train)

        # 评估拟合效果
        y_train_pred = LR_model.predict(self.X_train)
        y_test_pred = LR_model.predict(self.X_test)

        report = classification_report(self.y_train, y_train_pred)