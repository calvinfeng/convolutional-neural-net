from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.ndimage as ndimage
from img_data_utils import *


def main():
    model = ResNet50(weights='imagenet')
    feed_dict = load_jpg_from_dir("datasets/dog-vs-cat-train/", resize_px=224, num_images_per_class=1000, start_idx=1)
    preds = model.predict(feed_dict['X'])

    print 'Prediction shape %s' % str(preds.shape)

    for i, pred in enumerate(decode_predictions(preds, top=5)):
        list_of_categories = []
        for category in pred:
            list_of_categories.append(category[1])

        if feed_dict['y'][i] == 1:
            print '%s should be cat' % list_of_categories
        else:
            print '%s should be dog' % list_of_categories


if __name__ == '__main__':
    main()
