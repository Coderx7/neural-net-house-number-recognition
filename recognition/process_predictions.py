# Code hackily reproduced from example seen here:
# https://github.com/BVLC/caffe/blob/master/examples/detection.ipynb

# NOTE: The example code is only for a single image, so they use index 0 a lot.
# You have to manually change the index to get different bounding boxes.

# Currently unsure how to organize bounding boxes by image (e.g., image 0 has
# three bounding boxes, image 1 has 1, etc).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_hdf('det_output_OURMODEL.h5', 'df')
print(df.shape)
print(df.iloc[67])

with open('det_words_svhn.txt') as f:
    labels_df = pd.DataFrame([
        {
            'synset_id': l.strip().split(' ')[0],
            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
        }
        for l in f.readlines()
    ])
labels_df.sort('synset_id')
predictions_df = pd.DataFrame(np.vstack(df.prediction.values), columns=labels_df['name'])
print(predictions_df.iloc[67])

# max_s = predictions_df.max(0)
# max_s.sort(ascending=False)
# print(max_s[:10])

# # Find, print, and display the top detections: person and bicycle.
# i = predictions_df['Six'].argmax()
# j = predictions_df['Seven'].argmax()

# # Show top predictions for top detection.
# f = pd.Series(df['prediction'].iloc[i], index=labels_df['name'])
# print('Top detection:')
# print(f.order(ascending=False)[:5])
# print('')

# # Show top predictions for second-best detection.
# f = pd.Series(df['prediction'].iloc[j], index=labels_df['name'])
# print('Second-best detection:')
# print(f.order(ascending=False)[:5])

# # Show top detection in red, second-best top detection in blue.
# im = plt.imread('nums/109.png')
# plt.imshow(im)

# currentAxis = plt.gca()

# det = df.iloc[i]
# coords = (det['xmin'], det['ymin']), det['xmax'] - det['xmin'], det['ymax'] - det['ymin']
# print coords
# currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='r', linewidth=5))

# det = df.iloc[j]
# coords = (det['xmin'], det['ymin']), det['xmax'] - det['xmin'], det['ymax'] - det['ymin']
# currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='b', linewidth=5))

# plt.savefig("try_at_boxes.png")