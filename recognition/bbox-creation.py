# -*- coding: utf-8 -*-
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import glob

### From selectivesearch (https://github.com/AlpacaDB/selectivesearch)
### Modified by Alex King

# To use: this file expects .png images of the full SVHN set in a folder,
# currently called "nums". The code arbitrarily picked images 100.png ...
# 120.png, but you can edit the code to select whichever range of images you
# have on hand. It is currently configured to save images with bounding boxes
# in the nums directory, but that can be commented out.

# The key output here is bboxes.csv, a CSV file that can be fed into Caffe's
# detect.py function.

def main():

    csv = open("bboxes.csv", "w")
    csv.write("filename,ymin,xmin,ymax,xmax\n")
    images = glob.iglob('test/*.png')

    for i in images:

        # loading lena image
        img = skimage.io.imread(i)
        # img = skimage.data.astronaut()

        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(
            img, scale=500, sigma=0.9, min_size=10)

        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 20 pixels
            if r['size'] < 20:
                continue
            # distorted rects
            x, y, w, h = r['rect']

            if x is 0 or y is 0 or w is 0 or h is 0:
                continue
            if w / h > 1.5 or h / w > 1.5:
                continue
            candidates.add(r['rect'])

        # draw rectangles on the original image
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(img)
        for x, y, w, h in candidates:
            #print x, y, w, h
            csv.write('"nums/' + str(i) + '.png",' + str(y) + "," + str(x) + \
                      "," + str(y + h) + "," + str(x + w) + "\n")

            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

        plt.show()
        #plt.savefig("nums/" + str(i) + "boxes.png")
        plt.close()

if __name__ == "__main__":
    main()
