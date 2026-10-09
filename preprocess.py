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

# function to handle missing values in the data
def handle_missing_values(df):
    df = df[df.isna().any(axis=1)]
    return df



if __name__ == "__main__":
    print(df)