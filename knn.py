import pandas as pd
import matplotlib.pyplot as plt

from decisionTree import getData, setUp, splitTestAndTraining, evaluateModel

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# scaledata function because Knearest neighbour uses straight line (Euclidean) distances
# 
def scaleData(XTrain, XTest):
    # Setup scaler method from scikit-learn
    scaler = StandardScaler()
    XTrainScaled = scaler.fit_transform(XTrain)
    XTestScaled = scaler.transform(XTest)

    XTrainScaled = pd.DataFrame(XTrainScaled, columns=XTrain.columns, index=XTrain.index)
    XTestScaled = pd.DataFrame(XTestScaled, columns=XTest.columns, index=XTest.index)

    return XTrainScaled, XTestScaled





if __name__ == "__main__":
    df = getData()
    x, y, g = setUp(df)
    XTr, XTe, yTr, yTe = splitTestAndTraining(x, y, g)
    scaling = scaleData(XTr, XTe)
