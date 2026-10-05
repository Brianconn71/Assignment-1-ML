from pathlib import Path
import pandas as pd

# find the Data in the file system
DATA = Path(__file__).parent / "Data" / "Data.csv"

# Opening the file and reading its contents.
def openFile(dataObject):
    # Using pandas to read the csv file
    readFile = pd.read_csv(dataObject)
    # returning the file which can now be analysed.
    return print(readFile)

def describeData(file):
    for column in file:
        



        


if __name__== "__main__":
    openFile(DATA)
    describeData(DATA)
