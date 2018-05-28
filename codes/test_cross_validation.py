import pickle
import numpy as np
import os
from keras.models import load_model
from sklearn.model_selection import train_test_split
import codes.my_helper as helper


def confusion_matrix(image_val, label_val, model):
    # Get prediction from model
    pred_val = (model.predict(image_val))
    # print(pred_val.shape)
    # TP: wh-wh, FN: wh-bl, TN: bl-bl , FP: bl-wh
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    no_images = label_val.shape[0]
    rows = label_val.shape[1]
    cols = label_val.shape[2]

    for n in range(0, no_images):
        for i in range(0, rows):
            for j in range(0, cols):
                label = label_val[n][i][j]
                pred = pred_val[n][i][j]

                # True: white
                if label[1] != 0:
                    # Pred: white
                    if pred[1] > pred[0]:
                        tp += 1
                    # Pred: black
                    else:
                        fn += 1
                # True: black
                else:
                    # Pred: black
                    if pred[0] > pred[1]:
                        tn += 1
                    # Pred white
                    else:
                        fp += 1

    div = rows * cols * no_images
    print('TP:' + str(tp / div * 100) + '%')
    print('FN:' + str(fn / div * 100) + '%')
    print('FP:' + str(fp / div * 100) + '%')
    print('TN:' + str(tn / div * 100) + '%')


def main():
    # VALIDATION ON CAMVID DATASET
    images = helper.read_images('../datas/images_camvid/image/')
    size = (320, 256)
    images = helper.resize_images(images, size)
    images = np.array(images) / 255
    classes = [[192, 0, 128]]
    labels_path = "../datas/labels_transformed_camvid.p"
    if os.path.exists(labels_path):
        labels = pickle.load(open(labels_path, "rb"))
    else:
        labels = helper.read_images('../datas/images_camvid/label/')
        labels = helper.clear_label_colors(labels)
        labels = helper.resize_images(labels, size)
        labels = np.array(labels)
        labels = helper.rgb_to_classes(labels, classes)
        pickle.dump(labels, open(labels_path, "wb"))
    img_train, img_val, label_train, label_val = train_test_split(images, labels, test_size=0.15,
                                                                  shuffle=True, random_state=helper.random_state)

    print("CAMVID")
    model_path = '../models/roma_full.0001.500.h5'
    model = load_model(model_path)
    batch_size = 16
    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(model_path)
    print(score)
    confusion_matrix(img_val, label_val, model)

    model_path = '../models/roma_full.001.500.h5'
    model = load_model(model_path)
    batch_size = 16
    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(model_path)
    print(score)
    confusion_matrix(img_val, label_val, model)

    model_path = '../models/roma_full.freeze.001.500.h5'
    model = load_model(model_path)
    batch_size = 16
    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(model_path)
    print(score)
    confusion_matrix(img_val, label_val, model)

    model_path = '../models/camvid_full.500.h5'
    model = load_model(model_path)
    batch_size = 16
    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(model_path)
    print(score)
    confusion_matrix(img_val, label_val, model)

    # VALIDATION OR ROMA DATASET

    # list (116,) of (1024, 1280, 3) nparrays
    images = helper.read_images('../datas/images_roma/image/')
    size = (320, 256)
    images = helper.resize_images(images, size)
    images = np.array(images) / 255
    classes = [[255, 255, 255]]
    labels_path = "../datas/labels_transformed_roma.p"
    if os.path.exists(labels_path):
        labels = pickle.load(open(labels_path, "rb"))
    else:
        labels = helper.read_images('../datas/images_roma/label/')
        labels = helper.resize_images(labels, size)
        labels = np.array(labels)
        labels = helper.rgb_to_classes(labels, classes)
        pickle.dump(labels, open(labels_path, "wb"))
    img_train, img_val, label_train, label_val = train_test_split(images, labels, test_size=0.15,
                                                                  shuffle=True, random_state=helper.random_state)

    print("ROMA")
    model_path = '../models/roma_full.0001.500.h5'
    model = load_model(model_path)
    batch_size = 16
    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(model_path)
    print(score)
    confusion_matrix(img_val, label_val, model)

    model_path = '../models/roma_full.001.500.h5'
    model = load_model(model_path)
    batch_size = 16
    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(model_path)
    print(score)
    confusion_matrix(img_val, label_val, model)

    model_path = '../models/roma_full.freeze.001.500.h5'
    model = load_model(model_path)
    batch_size = 16
    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(model_path)
    print(score)
    confusion_matrix(img_val, label_val, model)

    model_path = '../models/camvid_full.500.h5'
    model = load_model(model_path)
    batch_size = 16
    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(model_path)
    print(score)
    confusion_matrix(img_val, label_val, model)

    img_val = images
    label_val = labels

    print("ROMA FULL")
    model_path = '../models/camvid_full.500.h5'
    model = load_model(model_path)
    batch_size = 16
    score = model.evaluate(img_val, label_val, batch_size=batch_size)
    print(model_path)
    print(score)
    confusion_matrix(img_val, label_val, model)


main()
