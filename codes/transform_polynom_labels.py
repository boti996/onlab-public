import pickle
import cv2
import numpy as np

import helpers as helper


# TODO. csúszóablakosan + break, amikor megtalálta a jobboldalit + ha nem találta, csúszóablak növelése
# Get coefficients from bitmap labels
def get_coeffs(labels):
    left = []
    right = []

    no_labels = len(labels)
    rows = len(labels[0])
    cols = len(labels[0][0])
    for n in range(0, no_labels):

        x_left = []
        y_left = []
        x_right = []
        y_right = []
        for row in range(0, rows):
            # Select every rows' firs color transitions -> left and right polynoms' dots.
            left_comes = True
            right_comes = True
            for col in range(0, cols):

                curr_pixel = labels[n][row][col]
                next_pixel = labels[n][row][min(col+1, cols-1)]
                # 100 is the lower treshold value for pixel intesity
                if curr_pixel > 100:
                    if left_comes:
                        x_left.append(row)
                        y_left.append(col)
                        left_comes = False
                    elif right_comes and curr_pixel > next_pixel:
                        x_right.append(row)
                        y_right.append(col)
                        right_comes = False

        left.append(np.polyfit(x_left, y_left, 2))
        right.append(np.polyfit(x_right, y_right, 2))
        print(str(n)+"/"+str(no_labels))

    return left, right

# Print the old bitmap type labels and the new bitmap images generated from the polynom type labels
# format of size: (row, col) !
def print_labels(left, right, size, labels, images):
    no_polys = len(left)
    for n in range(0, no_polys):

        curr_left = left[n]
        curr_right = right[n]
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
        cv2.imshow('new_label', new_label)
        cv2.imshow('old_label', labels[n])
        cv2.imshow('image', images[n])
        cv2.waitKey(0)
        # TODO: cv2.polylines()


def main():
    labels = pickle.load(open("../datas/images_mlnd/label/full_CNN_labels.p", "rb"))
    # labels = labels[:5]
    images = pickle.load(open('../datas/images_mlnd/image/my_coeff_images.p', "rb"))
    # images = np.array(images[:5])
    size = (320, 256)
    labels = helper.resize_images(labels, size)
    labels = np.array(labels)
    left, right = get_coeffs(labels)
    # Uncomment this line for showing results
    # print_labels(left, right, (256, 320), labels, images)

    new_labels = np.hstack((left, right))
    # Uncomment this line for saving labels
    # pickle.dump(new_labels, open('datas/images_mlnd/label/my_coeff_labels.p', "wb"))

main()
