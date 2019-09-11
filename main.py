import pandas as pd
import numpy as np
import sys

# load pandas dataframe

# check for valid filepath
if len(sys.argv) <= 1:
    print('Error: You must enter a filepath as command argument')
    exit()
else:
    print('Importing File: ' + sys.argv[1])
    filepath = sys.argv[1]

data = pd.read_csv(filepath)

print(data.describe())
