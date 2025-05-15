import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class DrawFigures:
    def __init__(self, df_train, X_train, y_train):
        self.df_train = df_train
        self.X_train = X_train
        self.y_train = y_train

    # 按pdf文件中的分组对自变量作图，观察每组中自变量间关系
    def drawAll(self):
        df_train = self.df_train
        X_train = self.X_train

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

    # 对两个自变量作出PCA线
    def drawPca_pair(self, X1: str ='X1', X2: str ='X2'):
        X_train = self.X_train
        df_train = self.df_train


        data_X1 = X_train[X1]
        data_X2 = X_train[X2]

        # Combine x1 and x2 into a single dataset
        X = np.vstack((data_X1, data_X2)).T
        # print(X)

        # fit PCA
        pca = PCA(n_components=2)
        pca.fit(X)

        components = pca.components_
        means = pca.mean_
        explained_variance = pca.explained_variance_
        covariance_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # plot data points
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes = axes.ravel()

        ax1 = axes[0]
        sns.stripplot(data=df_train, x=X1, y=X2, hue='y', jitter=True, alpha=1, ax=ax1)

        ax2 = axes[1]
        # plot full PCA lines and mark them
        ax2.scatter(X[:, 0], X[:, 1], alpha=0.6, edgecolor='k')
        ax2.set_xlabel(X1)
        ax2.set_ylabel(X2)
        ax2.axis('equal')

        for j, (length, vector) in enumerate(zip(explained_variance, components)):
            v = vector * 3 * np.sqrt(length)
            ax2.quiver(means[0], means[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.002)
            ax2.quiver(means[0], means[1], -v[0], -v[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.002)
            ax2.text(means[0] + v[0], means[1] + v[1], f'PC{j+1}', color='red', ha='center', va='center')

        # Print covariance matrix and eigenvalues/vectors in the corner of the subplot
        textstr = f'Cov Matrix:\n{np.array2string(covariance_matrix, precision=2)}\n'
        textstr += f'Eigenvalues:\n{np.array2string(eigenvalues, precision=2)}\n'
        textstr += f'Eigenvectors:\n{np.array2string(eigenvectors, precision=2)}'
        ax2.text(0.05, 0.05, textstr, transform=ax2.transAxes, fontsize=8,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        # 设置标题
        plt.suptitle(f"{X2}~{X1}", y=1.05)
        plt.show()
    
    # 选取两个自变量，作图并观察其间关系
    def drawPair(self):
        df_train = self.df_train
        X_train = self.X_train

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


if __name__ == "main":
    print("wwwwwww")