import csv
from pathlib import Path

# find the Data in the file system
DATA = Path(__file__).parent / "Data" / "Data.csv"

# Opening the file and reading its contents.
def openFile(dataObject):
    with open(dataObject, "r") as f:
        readFile = f.read()
    # returning the file which can now be analysed.
    return readFile


        


if __name__== "__main__":
    openFile(DATA)
