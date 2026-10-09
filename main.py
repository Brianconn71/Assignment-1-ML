from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# find the Data in the file system
DATA = Path(__file__).parent / "Data" / "Data.csv"

# Load the data
df = pd.read_csv(DATA)

# check the shape/size of the csv file
def checkData():
    # assigning a variable to the shape of the csv
    # returning this shape which will help us analyse the data
    # in this case (112516, 5) gets returned so we know how many rows and columns
    dataShape = df.shape
    return print(dataShape)

def checkMissingValues(data):
    # using isna() to go through the csv file to hightlight if there are missing values and sums the amount of missing values per columns
    missing = data.isna().sum()
    # returns a pandas dataframe with one column from the pandas series variable - missing
    return pd.DataFrame({"Missing": missing})

def processMissing(df):
    # show the rows which have missing data
    df = df[df.isna().any(axis=1)]

    # Remove these rows which we have found with missing values
    df = df.dropna()
    return df.shape

# Opening the file and reading its contents.
# def openFile(dataObject):
#     # Using pandas to read the csv file
#     readFile = pd.read_csv(dataObject)
#     # returning the file which can now be analysed.
#     return print(readFile)

# def describeData(file):
#     for column in file:

# Load the data and drop the 44 rows with no HR value

# df = df.dropna(subset=["HR"])

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
    checkData()
    print(processMissing(DATA))  



