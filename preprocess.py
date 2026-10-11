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

# Check the data shape - columns and amount of row values before we do any validating
def checkDataShapeBeforeValidating(df):
    print(df.shape)

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
    # empty values is assigned to the sum of all columns that have a missing values totaled together
    # hence the .sum().sum() sum per column and then sum per dataframe
    emptyValues = df.isnull().sum().sum()
    # if there still are empty values then print a warning
    # Otherwise let the user know there are no empty values
    if emptyValues:
        print(f"Warning: {emptyValues} empty values found in the dataset")
    else:
        print("No more empty values in Dataset")

def checkNoDuplicates(df):
    # Check the dataset to ensure that no two rows of values are exactly the same
    # follows similar logic as checkForNoEmptyValues
    duplicates = df.duplicated().sum().sum()
    # Given the Time (Sec) column it would be very difficult for their to be any duplicates and I dont want to remove this column
    # This is just to confirm there are no two rows the exact same.
    if duplicates:
        print(f"Warning: {duplicates} duplicate values found in the dataset")
    else:
        print("No duplicated values detected in the dataset")

def checkDataTypes(df):
    # Checking that the data in the columns are consistent.
    # mixed or text value data types would make pandas store it as a non numeric type
    # So, we need a check to make sure they are numeric.

    # assigns any columns which are non numeric to the variable nonNumeric
    nonNumeric = df.select_dtypes(exclude="number")
    # checks the lenght of the columns in the non numeric dataframe.
    # If there is a column thats non-numeric then len will be > 0
    # use this to print a warning message to the user.
    if len(nonNumeric.columns):
        columns = list(nonNumeric.columns)
        print(f"Warning: {columns} Non-Numeric columns found in the dataset")
    else:
        print("No Non-Numeric columns detected in the dataset")

# Check the data shape - columns and amount of row values after validation
def checkDataShapeAfterValidating(df):
    print(df.shape)

def returnCleanData():
    # This function acts as the conductor of the file, I will run this function from __main__ block
    # below which will run the necessary functions and clean the data. Making it ready to be worked 
    # with the ML algorithms provided.

    # Load the data from the file
    df = loadData()

    # Check dhape of data loaded from file
    checkDataShapeBeforeValidating(df)

    # CLean the Dataframe we have loaded above to remove missing vales
    df = handle_missing_values(df)

    # Now, check the Dataframe and ensure we are getting valid data such as,
    # The data has no more empty values, It has no duplicate values and the datatypes are consistent
    checkForNoEmptyValues(df)
    checkNoDuplicates(df)
    checkDataTypes(df)

    # Check Data Shape after validating the data
    checkDataShapeAfterValidating(df)

    # Now we can return the cleaned data
    return df




if __name__ == "__main__":
    returnCleanData()