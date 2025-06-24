print("Running summary.py...")

# Load libraries
from pandas import read_csv
# from pandas.plotting import scatter_matrix
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/chwoodv/MachineLearning/refs/heads/main/Data/hepatitis.csv"
names = ["Category","Age","Sex","ALB","ALP","ALT","AST","BIL","CHE","CHOL","CREA","GGT","PROT"]
dataset = read_csv(url, names=names)

# Summarize the dataset
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('Category').size())
 