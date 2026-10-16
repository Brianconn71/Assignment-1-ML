import pandas as pd
import matplotlib.pyplot as plt

from decisionTree import getData, setUp, splitTestAndTraining, evaluateModel

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# scaledata function because Knearest neighbour uses straight line (Euclidean) distances
def scaleData(XTrain, XTest):
    # Knn measures a straight line distance so requires this method as Trees only measure one feature at a time.
    # Setup scaler method from scikit-learn
    scaler = StandardScaler()
    # Fit transform learns mean and std devand then uses them.
    XTrainScaled = scaler.fit_transform(XTrain)
    # tranform uses the mean and standar deviation from above and uses it on the test set of data
    XTestScaled = scaler.transform(XTest)

    # Scaler returns a numpy array so i am transforming this to a dataframe which will have its original column names
    XTrainScaled = pd.DataFrame(XTrainScaled, columns=XTrain.columns, index=XTrain.index)
    XTestScaled = pd.DataFrame(XTestScaled, columns=XTest.columns, index=XTest.index)

    # return the scaled data
    return XTrainScaled, XTestScaled

def trainModel(XTrainScaled, yTrain, k):
    # using this to train the model
    # uniform weights to balance votes equally
    # n_neighbors k means to use the k input and chaeck k nearest rows columns.
    knn = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    # fit method stores the scaled training data in a cache like structure, knn is a lazy learner so real work only happens when prediciting
    knn.fit(XTrainScaled, yTrain)
    # using this to print the number for k which is used to train the KNN.
    print(f"K nearest neighbour model trained with k = {k}")
    # return the model for evaluation and plotting purposes.
    return knn

def visualiseKnnResults(model, XTestScaled, yTest):
     # Plot Confusion Matrix
    # Confusion Matrix Display essentially takes in same arguments as the confusionmatrix in function evaluatemodel and makes a coloured (Green) grid of results
    # Normalize added so as to make grid easier to read and understand by showing percentages. display labels shows the labels for the predictions either No or Yes
    # cmap="Greens" adds green shade to the boxes in grid so larger grid sections stand out.
    ConfusionMatrixDisplay.from_predictions(yTest, model.predict(XTestScaled), display_labels=["No", "Yes"], cmap="Blues", colorbar=False, normalize="true")
    # Plot the title of the Grid
    plt.title("KNN: Confusion Matrix of Test Set")
    # Stops overlaps in the diagram without then labels could be clipped and overlapping
    plt.tight_layout()
    # Save the grid as a png file in the figures folder in root.
    # dpi=160 is the resolution, higher value then gives a higher quality image
    plt.savefig("figures/confusionMatrixKNN.png", dpi=160)
    # Shows the diagram on the screen to the user.
    plt.show()


if __name__ == "__main__":
    df = getData()
    x, y, g = setUp(df)
    XTr, XTe, yTr, yTe = splitTestAndTraining(x, y, g)
    XTrainScaled, XTestScaled = scaleData(XTr, XTe)

    k = 80
    knnModel = trainModel(XTrainScaled, yTr, k)

    evaluateModel(knnModel, XTrainScaled, XTestScaled, yTr, yTe)
    visualiseKnnResults(knnModel, XTestScaled, yTe)
