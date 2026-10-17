"""
This file will compare the results from the two ML algorithms
"""
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from decisionTree import getData, setUp, splitTestAndTraining
from decisionTree import trainModel as trainTree
from knn import scaleData
from knn import trainModel as trainKnn


if __name__ == "__main__":
    # get the data
    df = getData()
    #setup the data
    X, y, groups = setUp(df)

    # split data to test and train datasets.
    XTr, XTe, yTr, yTe = splitTestAndTraining(X, y, groups)

    # Get decision tree trained
    tree = trainTree(XTr, yTr)

    # Scale the KNN
    XTrScaled, XTeScaled = scaleData(XTr, XTe)

    # Train the knn model
    knn = trainKnn(XTrScaled, yTr, 83)

    #Plot one figure with two panels one for Decision tree one for KNN
    fig, axis = plt.subplots(1, 2, figsize=(14, 6))

    ConfusionMatrixDisplay.from_predictions(
        yTe, tree.predict(XTe), display_labels=["No", "Yes"],
        normalize="true", cmap="Greens", im_kw={"vmin": 0, "vmax": 1},
        ax=axis[0]
    )
    axis[0].set_title("Decision Tree (test set)")

    ConfusionMatrixDisplay.from_predictions(
        yTe, knn.predict(XTeScaled), display_labels=["No", "Yes"],
        normalize="true", cmap="Reds", im_kw={"vmin": 0, "vmax": 1},
        ax=axis[1]
    )
    axis[1].set_title("KNN (test set), k= 51")

    fig.suptitle("Normalised Confusion Matrix")
    plt.tight_layout()
    plt.savefig("figures/compareConfusionMatrix.png", dpi=160)
    plt.show()
