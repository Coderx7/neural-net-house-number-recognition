# convert_mat_to_lmdb.py
# Modified by Alex King

# This file uses scipy and the pycaffe library to convert a .mat file
# (what SVHN provides) to LMDB, the filetype of choice for Caffe.

# Important: the output directory specified must be unique, or else
# data will be appended to existing files.

# Note: warnings are OK and may be ignored.

import scipy
import sys
import numpy as np
from scipy import io
import lmdb
import caffe
import random

def main():
    if len(sys.argv) != 3:
        print "Usage: python convert_mat_to_lmdb.py [file.mat] [output-directory]"
        return 1

    mat = io.loadmat(sys.argv[1])

    X = mat["X"]
    y = mat["y"]

    # Rearrange order such that dimensions are (N, C, H, W) where
    # N is number of images, C is channels (R, G, B), H is height and
    # w is width
    X = np.swapaxes(X, 0, 3)
    X = np.swapaxes(X, 1, 2)
    X = np.swapaxes(X, 2, 3)

    N = X.shape[0]

    map_size = X.nbytes * 10

    env = lmdb.open(sys.argv[2], map_size=map_size)

    print "Converting", sys.argv[1] + "..."

    with env.begin(write=True) as database:

        r = list(range(N))
        random.shuffle(r)

        for i in r:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tostring()
            datum.label = int(y[i]) - 1

            str_id = '{:08}'.format(i)

            database.put(str_id, datum.SerializeToString())

    env.close()

    print "Finished saving LMDB in directory", sys.argv[2]

    return 0

if __name__ == '__main__':
    main()