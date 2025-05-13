import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DrawFigures:
    def __init__(self):
        pass

    # 按pdf文件中的分组对自变量作图，观察每组中自变量间关系
    def drawAll(self, df_train, X_train):
        feature_columns = [col for col in X_train.columns if col.startswith('X')]
        # print(feature_columns)
        feature_dict = {}

        for i in range(0, len(feature_columns), 5):
            key = f"X{i+1} to X{i+6}"
            feature_dict[key] = feature_columns[i:i+5]
            feature_dict[key].append('y')
            g = sns.pairplot(df_train[feature_dict[key]], hue='y')
            g.fig.suptitle(key)
            plt.show()
        # sns.pairplot(df_train[], hue='y')
        # plt.show()

    # 选取两个自变量，作图并观察其间关系
    def drawPair(self, df_train, X_train):
        feature_columns = [col for col in X_train.columns if col.startswith('X')]

        # length = len(feature_columns)
        length = 3

        for i in range(length):
            for j in range(i+1, length):
                features = [feature_columns[i], feature_columns[j]]
                features.append('y')
                title = f"{feature_columns[i]} and {feature_columns[j]}"
                g = sns.pairplot(df_train[features], hue='y')
                g.fig.suptitle(title)
                plt.show