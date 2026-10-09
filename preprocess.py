"""
This file is used to pre process the csv file.
This means that this file willload the csv file. It will Check for missing values and decide what to do with them and will add any features to be used.
It will return a single clean processed DataFrame ready for use with the Machine Learning Algorithms.
"""
from pathlib import Path
import pandas as pd

# Build the path to the datafile
Data = Path(__file__).parent / "Data" / "Data.csv"

# Function to load the file.
def loadData():
    # Using Pandas to read the data file from the path
    df = pd.read_csv(Data)
    # return the dataframe
    return df

# function to handle missing values in the data
def handle_missing_values(df):
    # use the input argument dataframe - df
    # and check the rows which have empty values
    missing = df[df.isnull().any(axis=1)]
    # The below used for testing purposes to find and compare shape of dataframe before we remove values.
    # shape = df.shape
    # print(shape)
    # if the length of the dataframe with missing values is grerater than 0 indicating there are missing values
    if len(missing) > 0:
        # then drop these values from the dataframe
        # also, reset index as dropping values from the file will cause gaps in the index of the file contents
        # without it then pandas would add a new column index which I don't need.
        df = df.dropna().reset_index(drop=True)
    # return a clean dataframe
    return df

def checkForNoEmptyValues(df):
    emptyValues = df.isna().sum().sum()
    if emptyValues:
        print(f"Warning: {emptyValues} still remain in the dataset")
    else:
        print("No more empty values in Dataset")

def checkNoDuplicates():
    return

def checkDataTypes():
    return



if __name__ == "__main__":
    x = loadData()
    u = handle_missing_values(x)
    print(checkForNoEmptyValues(u))