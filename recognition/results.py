import csv
import string
import sys
from collections import defaultdict


''' expects tab separated predictions CSV to be of the format:
    filename "[p of 0, p of 1, ..., p of 9]" xmin ymin width height

    expect comma separated target CSV to be of the format:
    filename class xmin ymin width height
'''
def main():
    if len(sys.argv) != 3:
        print "Usage:"
        print "{0} <predictionCSV> <targetCSV>".format(sys.argv[0])
        return

    preds  = sys.argv[1]
    target = sys.argv[2]
    prep_preds = preprocess_preds(preds, .11)
    recall = get_recall(prep_preds, target)
    acc    = get_accuracy(prep_preds, target)
    print 'recall is: ', recall , ' and accuracy is: ', acc


''' returns accuracy for the given predictions using the given target csv '''
def get_accuracy(preds_csv, target_csv):
    num_correct = 0
    itp_preds  = multi_digit_preds_from_csv(preds_csv)
    itp_target = multi_digit_preds_from_csv(target_csv)

    for img in itp_preds:
        if itp_preds[img] == itp_target[img]:
            num_correct += 1

    return float(num_correct) / len(itp_preds)


''' used to determine which digit appears first in a number '''
def bbox_compare(b1, b2):
    x1, y1 = b1[0], b1[1]
    x2, y2 = b2[0], b2[1]
    if x1 < x2:
        return 1
    elif x1 == x2:
        if y1 < y1:
            return 1
        elif y1 == y2:
            return 0
        else:
            return -1
    else:
        return -1


''' returns a dictionary containing image names and their corresponding
    classes for the given csv
'''
def multi_digit_preds_from_csv(csv_preds):
    img_to_class = dict()
    # images to list of preds for boudning boxes
    img_to_bbox_preds = defaultdict(list)

    with open(csv_preds, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # skip header
        next(reader, None)
        for row in reader:
            # map image name to list of (pred, [left, top, width, height])
            bbox = [int(x) for x in row[2:]]
            img_to_bbox_preds[row[0]].append((row[1], bbox))

        # concatinate the single digit predictions together
        # assuming the left most upper most digit comes first
        for img in img_to_bbox_preds:
            bbox_preds = img_to_bbox_preds[img]
            ordered_digits = [x for x, bbox in sorted(bbox_preds, reverse=True, cmp=bbox_compare, key=lambda p: p[1])]
            img_to_class[img] = int(''.join(ordered_digits))

    return img_to_class


''' returns the recall for the given predictions with the given target'''
def get_recall(preds_csv, target_csv):
    digits_correct = 0
    total = 0

    with open(preds_csv, 'r') as pf, open(target_csv) as tf:
        pr = csv.reader(pf, delimiter=',')
        tr = csv.reader(tf, delimiter=',')
        # skip headers
        next(pr, None)
        next(tr, None)

        # stores the digits present in the given image under the .png name
        img_to_digits = defaultdict(list)
        for row in tr:
            img_to_digits[row[0]].append(int(row[1]))

        # calc recall
        images = set()
        for row in pr:
            img_name = row[0]
            pred     = int(row[1])
            if pred in img_to_digits[img_name]:
                digits_correct += 1
            images.add(img_name)

        # get the total number of digits from the images for which we have preds
        total = sum([len(img_to_digits[img]) for img in images])

    return float(digits_correct) / total


''' creates a cvs with the prefix prep containing predictions are above percent conf '''
def preprocess_preds(preds, percent_conf):
    with open(preds, 'r+') as f, open('prep_' + preds, 'w') as pf:
        reader = csv.reader(f, delimiter='\t')
        prep_preds = [next(reader, None)]
        for row in reader:
            scores = [float(x) for x in row[1].strip(string.punctuation).split()]
            pred, score = max(enumerate(scores), key=lambda x: x[1])
            if score >= percent_conf:
                prep_row = [row[0][17:], pred, int(row[2]), int(row[3]), int(row[4]), int(row[5])]
                prep_preds.append(prep_row)

        writer = csv.writer(pf, delimiter=',')
        for row in prep_preds:
            writer.writerow(row)

    return 'prep_' + preds

main()


