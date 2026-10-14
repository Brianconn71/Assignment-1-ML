"""
This file takes in the data which has been preprocessed from the preprocess.py file and runs a Decision Tree 
Machine Learning Algoritm on it.
"""
from preprocess import returnCleanData
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
    # split we are aiming for, test_size=0.2 ensures we are splitting data based on 80:20 with 20% of participants going to testing,
    # random_state=25 - ensures we get the same split each time we run the algorithm, any number would work so 25 chosen at random. 
    splitData = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=25)
    # This is used to take the split data variable and use it on the dataset.
    # it will return a generator which produces results one at a time. The split is going to be 
    # Numpy arrays which will contain splits for test and training data.
    trainIdx, testIdx = next(splitData.split(X, y, groups))
    # below two lin use the iloc method to select a row based on its position.abs@ since we are taking in an array of positions it gives us just those rows to use.
    # So, XTrain holds 80% of participants for training and XTest 20% for testing.
    # yTrain and Test are the same so that the data doesnt get mixed up and cause misalignment issues
    XTrain, XTest = X.iloc[trainIdx], X.iloc[testIdx]
    yTrain, yTest = y.iloc[trainIdx], y.iloc[testIdx]

    # Return the four variables for use in training and testing
    return XTrain, XTest, yTrain, yTest

# this method just prints to the terminal to ensure we have correctly setup the training and test variables.
def checkTrainAndTestValues(XTrain, XTest):
    print("Training Data:", len(XTrain))
    print("Testing Data:", len(XTest))
    # print("Test participants:", sorted(groups.iloc[testIdx].unique().tolist()))

# Using this function to build and train the decision tree classifoer in scikit-learn.
# This function only takes training data from the splitTestAndTrain Fucntion and explicitly avoids the test data
# the whole point is to avoid leaking data which would happen if we included test data
def trainModel(XTrain, yTrain):
    # Create the decision tree with chosen settings
    # max_depth = 5 is to avoid overfitting, basically at 5 we are limiting how complex the decision tree becomes, higher value will mean a more complex tree which may better fit but it will memorise noise
    # random_state=25 ensures results are reproducible, if two pieces of data are the same then it will remember which way to split the data
    # class_weight="balanced" used to set the weight of each class weight to total rows. 
    # criterion = "gini" measures how well the split of the data is, so if p=1 then Gini - 1-1 =0 so it would be pure and 50/50 mix is what we wnat so 
    # Gini = 1 - (0.25 +0.25) =0.5
    decisionTree = DecisionTreeClassifier(max_depth=5, random_state=25, class_weight="balanced", criterion="gini")
    # where the training of the data happens based on the training data from the split we got in splitTestAndTraining function above
    # Starts with all of the training data, then searches every feature for the split that would best reduce ginin index
    # then splits rows into two branches and repeats process again, stops when max_depth is reached or can't be split any further.
    # finally it will label each row and that is then the prediciton for any new row of data which may be similar to it.
    trainedTree = decisionTree.fit(XTrain, yTrain)
    # below two prints statements check on the size of the tree.
    # gets how many levels deep the decision tree reached
    print(trainedTree.get_depth())
    # gets the number of leaves the tree has reached
    # A tree with max_depth = 5 means at most 2(5) = 32 leaves maximum
    print(trainedTree.get_n_leaves())

    # Now return the trained model for use in evaluateModel to predict the test data on it.
    return trainedTree

# evalutaeModel method which will take the trained model we return in trainModel and evaluates the test data based on this trained model
def evaluateModel(trainedModel, XTrain, XTest, yTrain, yTest):
    # Sends every training row of data down the trained model, essentially asking 'is HR > than? and is Respr > than?'
    # until it reaches the bottom leaf of the tree which gives us our answer. The result is an array of predictions
    trainPrediction = trainedModel.predict(XTrain)
    # Does the exact same job as above but on the test data rather than training data. 
    testPrediction = trainedModel.predict(XTest)

    # printing to terminal with results
    print("_____ Training Set _____")
    # This compares the training labels with the model predictions, it prints the precision, recall F1 and support values
    print(classification_report(yTrain, trainPrediction, target_names=["No", "Yes"]))
    # prints a 2 x 2 table with counts 
    print(confusion_matrix(yTrain, trainPrediction))
    """
    accuracy_score(trainPrediction, testPrediction)
    precision_score(trainPrediction, testPrediction)
    recall_score(trainPrediction, testPrediction)
    f1_score(trainPrediction, testPrediction)
    """

    # All three lines below follow the same logic as training set above and compares can be made
    print("_____ Test Set _____")
    print(classification_report(yTest, testPrediction, target_names=["No", "Yes"]))
    print(confusion_matrix(yTest, testPrediction))

    """accuracy_score(trainPrediction, yTest)
    precision_score(trainPrediction, yTest)
    recall_score(trainPrediction, yTest)
    f1_score(trainPrediction, yTest)"""

    return

# This function is used to plot the results of the evaluatemodel to the user.
def visualiseResults():
    # Plot the 
    return


if __name__ == "__main__":
    df = getData()
    x, y, g = setUp(df)
    XTr, XTe, yTr, yTe = splitTestAndTraining(x, y, g)
    trainedTree = trainModel(XTr, yTr)
    evaluateModel(trainedTree, XTr, XTe, yTr, yTe)

    