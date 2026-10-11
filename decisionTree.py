"""
This file takes in the data which has been preprocessed from the preprocess.py file and runs a Decision Tree 
Machine Learning Algoritm on it.
"""
from preprocess import returnCleanData
from sklearn.model_selection import GroupShuffleSplit

# Taking the cleaned dataset from preprocess and assinging to the df variable.
def getData():
    df = returnCleanData()
    return df

# Sci Kit learn needs needs to know the features of the dataframe we wish to train
# along with the target which is what we want to make our prediciton based on.
# also, in this casewhat we are using to group our data
def setUp():
    # X is the features we are using, The features which the ML model will learn from
    # in our case it will be the HR (Heart rate) column and the respr(respiration) columns
    # scikit-Learn expects a dataframe which is what the double brackets achieves. 
    # outer brackets to select from the dataframe and inner to return as a list of names
    # convention dictates x be in caps.
    X = df[["HR", "respr"]]
    y = df["Label"]
    groups = df["Participant"]
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
    