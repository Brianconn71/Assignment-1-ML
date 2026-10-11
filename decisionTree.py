"""
This file takes in the data which has been preprocessed from the preprocess.py file and runs a Decision Tree 
Machine Learning Algoritm on it.
"""
from preprocess import returnCleanData
from sklearn.model_selection import GroupShuffleSplit

# Taking the cleaned dataset from preprocess and assinging to the df variable.
df = returnCleanData()

def setUp():
    return

def splitTestAndTraining():
    return

def trainModel():
    return

def evaluateModel():
    return

def visualiseResults():
    return

# Features (X), target (y), and the groups to split by
# X = df[["HR", "respr"]]
# y = df["Label"]
# groups = df["Participant"]

# 80:20 split, keeping each participant entirely in train or test
# splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, test_idx = next(splitter.split(X, y, groups))

# X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
# y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# print("Training rows:", len(X_train))
# print("Test rows:", len(X_test))
# print("Test participants:", sorted(groups.iloc[test_idx].unique().tolist()))

if __name__ == "__main__":
    print(df)
    