# Code hackily reproduced from example seen here:
# https://github.com/BVLC/caffe/blob/master/examples/detection.ipynb

# NOTE: The example code is only for a single image, so they use index 0 a lot.
# You have to manually change the index to get different bounding boxes.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import StringIO
import csv

# Load our model predictions
df = pd.read_hdf('det_output_OURMODEL.h5', 'df')

# See link for what this does; basically, that file maps indices of classes
# to strings, e.g. class 0 is "One", class 9 is "Zero"
# (It's off by one based on how we generate data, we can fix that later)
with open('det_words_svhn.txt') as f:
    labels_df = pd.DataFrame([
        {
            'synset_id': l.strip().split(' ')[0],
            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
        }
        for l in f.readlines()
    ])
labels_df.sort('synset_id')

# Go this route: Get access to prediction data for each filename.
# Downside: predictions_df ignores bounding boxes.
names = df.prediction.index.tolist()
names_no_dups = list(set(names))
predictions_df = pd.DataFrame(np.vstack(df.prediction.values), 
                              columns=labels_df['name'])
predictions_df.index = names

for name in names_no_dups:
    print name
    maxer = predictions_df.loc[name]
    print maxer.max()
    print

# Alternative: Immediately save the entire thing to CSV, then open and re-parse
# In more friendly structures.
df.to_csv("data_frame.csv", sep='\t')
