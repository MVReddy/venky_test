import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Quality(object):
    def __init__(self, data_set_file):
        self.data_file_name = data_set_file
        self.data = None
        self.white_wine = None
        self.validation_set = None
        self.test_set = None
        self.training_set = None

    def read_data(self):
        self.data = pd.read_csv(self.data_file_name, delimiter=';')
        return self.data

    def task1(self):
        df = self.data.copy()
        if not df.empty:
            quality_data_set = list(df.quality.unique())
            print("data set Info: \n")
            print(df.info())
            print("Number of DataPoints(rows) in dataset: {}".format(len(df)))
            print("Number of Features(columns) in dataset: {}".format(len(df.columns)))
            print("All classes in dataset are: {}".format(quality_data_set))
            print("Number of data points in each class: ")
            print("CLASS \t\t TOTAL DATA POINTS")
            for item in quality_data_set:
                print(item, "\t\t\t", len(df[df.quality==item]))

            self.white_wine = shuffle(df, random_state=0)
            ax1 = df.plot.scatter(x='fixed acidity', y='residual sugar', c='DarkBlue')
            # ax1.draw()

    def task2(self):
        pca = PCA(n_components=5)
        pca.fit(self.white_wine)
        print(pca.components_)
        N = len(self.white_wine)
        x = self.white_wine['fixed acidity']
        y = self.white_wine['residual sugar']
        colors = np.random.rand(N)
        area = (30 * np.random.rand(N)) ** 2
        plt.scatter(x=x, y=y, s=area, c=colors, alpha=0.5)
        plt.show()
        print("*"*30)
        print("VARIANCE OF EACH COMPONENT")
        print("*" * 30)
        print(self.white_wine.var())

    def task3(self):
        self.white_wine = self.white_wine.reset_index()
        self.validation_set = self.white_wine.iloc[:1000]
        self.test_set = self.white_wine.tail(1000)
        self.training_set = self.white_wine[1001:self.test_set.index.values[0]-1]

    def task4(self):
        X = self.training_set['density'].values[:20]
        y = self.training_set['sulphates'].values[:20]

        # Reshaping data set
        X = X.reshape(20, 1)
        y = y.reshape(20, 1)

        # Fitting Simple Linear Regression to the set
        regressor = LinearRegression()
        regressor.fit(X, y)
        # Predicting
        print(regressor.predict(X))
        # Visualising the set results
        plt.scatter(X, y, color='red')
        plt.plot(X, regressor.predict(X), color='blue')
        plt.title('density vs sulphates')
        plt.xlabel('density')
        plt.ylabel('sulphates')
        plt.show()


if __name__ == '__main__':
    obj = Quality('winequality-white.csv')
    df = obj.read_data()
    obj.task1()
    obj.task2()
    obj.task3()
    obj.task4()
