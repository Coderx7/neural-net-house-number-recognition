Jason Krone and Alex King
COMP 135 Final Project

Neural Network Street View House Number (SVHN) Recognition

Update 4/8:

Digit recognition and classification is now demoable.

### NOTE: These files are not yet very well organized. "svhn_model" contains 
certain files that should be used in the directory where an SVHN model is trained,
(such as the cifar10 example directory), and "recognition" contains some files and scripts
associated with the recognition task outlined below.

Also note that because the image data is around 300mb, it is not hosted here, so it must
be generated with convert_mat_to_lmdb.py.

- Prerequisite: A .caffemodel file trained on the SVHN dataset. You can use the file titled "SVHN_9144acc_quick_5000.caffemodel.h5" or generate it yourself: Use the cifar10
model, but with our data and our mean image binaryproto file.
- Follow the general outline here: https://github.com/BVLC/caffe/blob/master/examples/detection.ipynb
- Instead of the referenced Matlab selective search module, we'll use
"selectivesearch". https://github.com/AlpacaDB/selectivesearch (Follow instructions
to install with pip)
- Because we aren't using their intended selective search module, you have to
pass in a CSV of image files and bounding boxes. Use the bbox-creation script
in the recognition folder to create the file and (optionally) bounding box images.
- Caffe's detector.py (in caffe/python/caffe/detector.py) requires a small edit.
See "detector.py" for what should be changed. Make that change in your Caffe installation before running.

- Create a _temp directory in the Caffe directory, as suggested by the notebook linked above.
You need many of the files in our "recognition" folder to go here.

The following command:

../python/detect.py --crop_mode=list --pretrained_model=../examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5 --model_def=../examples/cifar10/cifar10_quick.prototxt --gpu --raw_scale=255 --context_pad=5 bboxes.csv det_output_OURMODEL.h5

Does a few things:

- Run "detect" (what we want) in "list" mode (what we need to use). 
- It points to a model created after 5000 training iterations. It points to a model definition.
This example has "cifar10", but make sure it's pointing towards your svhn training
directory files. 
- Sets "context_pad" to 5. The default of 16 caused a DivisionByZero error because of
a coincidence with our image size (32x32).
- Saves output to det_output_OURMODEL.h5. This can be renamed.

It should run very quickly on a g2.2 EC2 instance and save that file. You can
then inspect it with the (rough) process_predictions.py.

### Where we go from here:

- We would like a more structured, versatile Python program to iterate over the
entire fullsize image folder, create the suggested bbox csv file, and also read
the dataset's labels/bboxes for storage in another file.
- We would like some basic way to determine how well we're doing at recognition+classification. That involves comparing *all bounding boxes and all labels* of our data with *all bounding boxes
and all labels* provided to us in the full SVHN dataset.
- We would like higher than 91% accuracy in our digit classifier.
- We would like a neat folder architecture such that the entire git repository
can be cloned directly into Caffe ready to run.


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
