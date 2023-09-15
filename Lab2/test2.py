import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')
iris = load_iris()
iris = sns.load_dataset('iris')
iris.head()
iris_setosa = iris.loc[iris["species"] == "Iris-setosa"]
iris_virginica = iris.loc[iris["species"] == "Iris-virginica"]
iris_versicolor = iris.loc[iris["species"] == "Iris-versicolor"]

sns.FacetGrid(iris,
              hue="species",
              ).map(sns.distplot,
                          "petal_length").add_legend()
sns.FacetGrid(iris,
              hue="species",
              ).map(sns.distplot,
                          "petal_width").add_legend()
sns.FacetGrid(iris,
              hue="species",
             ).map(sns.distplot,
                          "sepal_length").add_legend()
plt.show()
