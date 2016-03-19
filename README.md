Jason Krone and Alex King
COMP 135 Final Project

Neural Network Street View House Number (SVHN) Recognition

Notes 3/19:

The SVHN dataset comes in two different formats, so we'll need two different
scripts to convert each into a format that Caffe can use. The 32x32 sets are
stored as .mat files, while the full-image sets are stored as PNGs with bounding
box information stored in a parallel .mat file.

Caffe can read LMDB files, and it can also convert folders of images to LMDB
files. For the 32x32 datasets, it makes the most sense to convert them directly
to LMDB. For the full images, it will probably be easier to organize the images
into subfolders and let Caffe do the conversion.

As of 3/19, 32x32 .mat -> LMDB conversion is working.

Next Steps:

- Experiment running a Caffe model on the 32x32 datasets, now that we have them in the proper
format.
- Write a Python script to convert the full image datasets to a folder hierarchy
understandable by Caffe (see documentation).
- Create a strong Caffe model to classify 32x32 digits. (Part A)
- Create a strong Caffe model to establish a bounding box around all digits
located on an image. (Part B)
- Connect our two models such that after bounding boxes are identified in Part B,
our classifier in Part A can classify them.