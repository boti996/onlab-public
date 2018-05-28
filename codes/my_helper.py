import os
import cv2
import numpy as np
import keras.preprocessing.image as im
from keras.backend import epsilon

import helpers as helper_old


# random state value
random_state = 1234

mlnd_norm = np.array([10**5, 10**2, 10**-2, 10**5, 10**1, 10**-2], dtype=np.float32)
mlnd_denorm = np.array([10**-5, 10**-2, 10**2, 10**-5, 10**-1, 10**2], dtype=np.float32)


# Read images from parameter folder
# folder_url must end with '/'
def read_images(folder_url, param=None):
    images = []
    files = os.listdir(folder_url)
    files.sort()
    for image_file in files:
        if param:
            image = cv2.imread(folder_url + image_file, param)
        else:
            image = cv2.imread(folder_url + image_file)
        images.append(image)
    return images


# Resize a list of images
# size: (width, height)
def resize_images(images, size):
    for i in range(0, len(images)):
        images[i] = cv2.resize(images[i], size)
    return images


# Delete unused color-markings from CamVid labels
def clear_label_colors(labels):
    # BGR color-channel mode !
    lane_markings_driv_upper = np.array([192, 0, 128])
    lane_markings_driv_lower = np.array([192, 0, 128])
    i = 0
    for label in labels:
        mask = cv2.inRange(label, lane_markings_driv_lower, lane_markings_driv_upper)
        label = cv2.bitwise_and(label, label, mask=mask)
        labels[i] = label
        i += 1
    return labels


# Default color class
default_class = [[255, 255, 255]]


# Transform rgb images to pixel-per-pixel, one-hot classified images
# classes structure: class0 = [0, 0, 0],  (class1, ..., classN): classes=[[r,g,b], ..., [r,g,b]]
# size of output images: [height][width][len(classes)+1]
def rgb_to_classes(labels, classes=default_class):
    shape = labels.shape
    class_number = len(classes)
    labels_new = np.zeros((shape[0], shape[1], shape[2], class_number + 1), dtype=np.uint8)
    for n in range(0, shape[0]):
        non_zero = 0
        for i in range(0, shape[1]):
            for j in range(0, shape[2]):
                if np.array_equal(labels[n][i][j], np.array([0, 0, 0])):
                    labels_new[n][i][j][0] = 1
                else:
                    for c in range(0, class_number):
                        if np.array_equal(labels[n][i][j], classes[c]):
                            labels_new[n][i][j][c + 1] = 1
                            non_zero += 1
                            break
        print('#' + str(n) + ' non-zero: ' + str(non_zero))
    return labels_new


# Transform rgb images to pixel-per-pixel, binary classified images
# size of output images: [height][width][1]
def rgb_to_binary(labels, classes=default_class):
    shape = labels.shape
    labels_new = np.zeros((shape[0], shape[1], shape[2], 1), dtype=np.uint8)
    for n in range(0, shape[0]):
        non_zero = 0
        for i in range(0, shape[1]):
            for j in range(0, shape[2]):
                # Black pixel
                if np.array_equal(labels[n][i][j], np.array([0, 0, 0])):
                    labels_new[n][i][j][0] = 0
                else:
                    # Colored pixel
                    if np.array_equal(labels[n][i][j], classes[0]):
                        labels_new[n][i][j][0] = 1
        print('#' + str(n) + ' non-zero: ' + str(non_zero))
    return labels_new


# Blend images with labels or with outputs predicted by model
# Use with a compiled model or with a labels array
# folder_url must end with '/'
def blend_images(images, folder_url, labels=None, model=None, classes=default_class, samplewise_std_normalization=False, samplewise_center=False):
    for i in range(0, len(images)):
        img = images[i]

        if model is not None:
            if samplewise_std_normalization or samplewise_center:
                img_norm = np.copy(img)
                img_norm -= np.mean(img_norm, keepdims=True)
                if samplewise_std_normalization:
                    img_norm /= (np.std(img_norm, keepdims=True) + epsilon())
                prediction = (model.predict(np.array([img_norm, ])))[0]
            else:
                prediction = (model.predict(np.array([img, ])))[0]

        elif labels is not None:
            prediction = labels[i]
        else:
            raise Exception("blend_images: Add a valid model or labels array")
        prediction = classes_to_rgb(prediction, classes)
        prediction = white_recolor(prediction)

        # alpha = 1
        # beta = 1
        alpha = 0.2
        beta = 1 - alpha
        img = (img * 255).astype(np.uint8)
        blended = cv2.addWeighted(img, alpha, prediction, beta, 0.0)

        cv2.imwrite(folder_url + str(i) + '.jpeg', blended)


# blend images with labels or with outputs predicted by model
# use with a compiled model or with a labels array
# folderUrl: destination folder; string with '/' at the end
def blend_images_polynom(images, folder_url, labels=None, model=None):
    size = images[0].shape
    no_images = len(images)
    for n in range(0, no_images):
        image = images[n]
        prediction = (model.predict(np.array([image, ])))[0]
        prediction = np.multiply(prediction, mlnd_denorm)
        left = prediction[0:3]
        right = prediction[3:]

        curr_left = left
        curr_right = right
        x_fit = np.linspace(0, size[0] - 1, size[0])
        y_fit_left = curr_left[0] * x_fit**2 + curr_left[1] * x_fit + curr_left[2]
        y_fit_right = curr_right[0] * x_fit**2 + curr_right[1] * x_fit + curr_right[2]

        y_cross = 0
        for i in range(0, size[0]):
            if y_fit_left[i] <= y_fit_right[i]:
                y_cross = i
                break
        if y_cross > 255:
            y_cross = 255
        elif y_cross < 0:
            y_cross = 0

        y_fit_left = y_fit_left[y_cross:]
        y_fit_right = y_fit_right[y_cross:]
        x_fit = x_fit[y_cross:]

        zeros = np.zeros(shape=(size[0], size[1])).astype(np.uint8)
        new_label = np.dstack((zeros, zeros, zeros))

        pts_left = np.array([np.transpose(np.vstack([y_fit_left, x_fit]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([y_fit_right, x_fit])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(img=new_label, pts=np.int_([pts]), color=(255, 255, 255))

        alpha = 1
        beta = 1
        image = (image * 255).astype(np.uint8)
        blended = cv2.addWeighted(image, alpha, new_label, beta, 0.0)

        cv2.imwrite(folder_url + str(n) + '.jpeg', blended)


# change white pixels of label image to blue (BGR color mode !)
def white_recolor(image):
    for r in image:
        for c in r:
            if c[2] > 100:
                c[2] = 0
    for r in image:
        for c in r:
            if c[1] > 100:
                c[1] = 0
    return image


# Transform per-pixel classified image into rgb images
# Output size: [height][width][3]
# TODO: one-hot vectoros formánál csak ki kéne venni a [][][1] -es réteget
def classes_to_rgb(image, classes=default_class):
    shape = image.shape
    pred_new = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    non_zero = 0
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            c = max_prob(image[i][j])
            if c == 0:
                pred_new[i][j] = np.array([0, 0, 0])
            else:
                pred_new[i][j] = classes[c - 1]
                non_zero += 1
    print("NON_ZERO: " + str(non_zero))
    return pred_new


def max_prob(pred_pixel):
    max_probability = 0
    max_i = 0
    for i in range(0, len(pred_pixel)):
        if pred_pixel[i] > max_probability:
            max_i = i
            max_probability = pred_pixel[i]
    return max_i


def transform_batch(image_batch, label_batch, horizontal_flip=False, rotation_range=0., channel_shift_range=0.,
                    samplewise_std_normalization=False, samplewise_center=False):

    shape = image_batch.shape
    img_channel_index = 2
    img_row_index = 0
    img_col_index = 1
    if shape[1] > shape[2]:
        img_row_index = 1
        img_col_index = 0

    length = len(image_batch)
    for i in range(0, length):
        curr_image = image_batch[i]
        curr_label = label_batch[i]

        # Horizontal flip
        if horizontal_flip:
            if np.random.random() < 0.5:
                axis = img_col_index
                curr_image = np.asarray(curr_image).swapaxes(axis, 0)
                curr_image = curr_image[::-1, ...]
                curr_image = curr_image.swapaxes(0, axis)

                curr_label = np.asarray(curr_label).swapaxes(axis, 0)
                curr_label = curr_label[::-1, ...]
                curr_label = curr_label.swapaxes(0, axis)

        # Rotation
        if rotation_range != 0.:
            theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])

            curr_label = im.apply_transform(curr_label, rotation_matrix, channel_axis=img_channel_index)
            curr_image = im.apply_transform(curr_image, rotation_matrix, channel_axis=img_channel_index)

        # Channel shift
        if channel_shift_range != 0.:
            curr_image = im.random_channel_shift(curr_image, channel_shift_range, channel_axis=img_channel_index)

        # Normalization
        if samplewise_std_normalization:
            if not samplewise_center:
                samplewise_center = True

        if samplewise_center:
            curr_image -= np.mean(curr_image, keepdims=True)

        if samplewise_std_normalization:
            curr_image /= (np.std(curr_image, keepdims=True) + epsilon())

        image_batch[i] = curr_image
        label_batch[i] = curr_label

    # blend_images(images=image_batch, labels=label_batch, folder_url='../datas/images_camvid/output/')

    return image_batch, label_batch
