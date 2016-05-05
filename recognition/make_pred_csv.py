import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import StringIO
import csv

# Load our model predictions
df = pd.read_hdf('det_output_OURMODEL.h5', 'df')
df.to_csv("data_frame.csv", sep='\t')


