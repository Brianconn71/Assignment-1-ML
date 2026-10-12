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
# also, in this case what we are using to group our data, this is being used when using Scikit learn method GroupShuffleSplit
# To split the data based on an 80:20 training testing split.
def setUp(df):
    # X is the features we are using, The features which the ML model will learn from
    # in our case it will be the HR (Heart rate) column and the respr(respiration) columns
    # scikit-Learn expects a dataframe which is what the double brackets achieves. 
    # outer brackets to select from the dataframe and inner to return as a list of names
    # convention dictates x be in caps.
    X = df[["HR", "respr"]]
    # y is the answer we are looking for the algorithm to predit, 0 or 1 in this case
    # y is the convention used and will store a pandas series.
    y = df["Label"]
    # groups isnt a feature its just used to help group the data based on participants in this case.
    # Its aim will be to keep participants data together for testing and training purposes.
    groups = df["Participant"]

    return X, y, groups

# We need to splt the data into training data and testing data. This method achieves this.
# We are splitting data based on an 80:20 split which means 80% of the data will be used to train the model,
# and 20% will be used to test the model. It also will aim to keep each of the participants in the dataset, 
# which we found in setup() in either Testing or Training without overlap.
def splitTestAndTraining(X, y, groups):
    # This is used to create a splitter object - GroupShuffleSplit is a Scikit-Learn method which I am using to 
    # use the groups from setup() and split them fully into either training or testing. n_splits is set to 1 meaning we only need one split based on the 80:20
    # split we are aiming for, test_size=0.2 ensures we are splitting data based on 80:20 with 20% going to testing,
    # random_state=25 - ensures we get the same split each time we run the algorithm, any number would work so 25 chosen at random. 
    splitData = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=25)
    # This is used to take the split data variable and use it on the dataset.
    # it will return a generator which produces results one at a time. The split is going to be 
    # Numpy arrays which will contain splits for test and training data.
    train, test = next(splitData.split(X, y, groups))

def trainModel():
    return

def evaluateModel():
    return

def visualiseResults():
    return

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
    