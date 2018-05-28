import codes.my_helper as helper
import cv2
import numpy as np
import random


# TODO: megcsinálni

def create_skeleton(labels):
    skeletons = []
    for img in labels:
        # shape: (256, 320)
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        ret, img = cv2.threshold(img, 127, 255, 0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        while (not done):
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True
        skeletons.append(skel)
        # cv2.imshow("skel", skel)
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return skeletons


def print_labels(left, right, labels, size):
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
        cv2.imshow('old label', labels[n])
        cv2.waitKey(0)
        # TODO: cv2.polylines()-t kipróbálni


def create_polynoms(skeletons):
    rows, cols = skeletons[0].shape[0], skeletons[0].shape[1]

    start_col = cols // 2
    start_row = rows // 3
    start_center_dist = cols // 6
    max_center_dist = cols // 3

    left, right = [], []
    # for img in skeletons:
    # img = skeletons[3]
    for img in skeletons:
        found_left, found_right = False, False
        x_left, y_left, x_right, y_right = [], [], [], []

        for row in range(start_row, rows):
            curr_max_center_dist = start_center_dist + (max_center_dist - start_center_dist) * (row - start_row) // (
                        rows - start_row)
            for col_offset in range(0, curr_max_center_dist):
                # print(str(row) + ' ' + str(curr_max_center_dist) + ' ' + str(start_col - col_offset))
                if img[row][start_col - col_offset] != 0:
                    found_left = True
                    print("left found")
                    x_left.append(row)
                    y_left.append(start_col - col_offset)
            if found_left:
                break

        if found_left:
            i = 0
            for row in range(x_left[0] + 1, rows):
                col_center = y_left[i]
                for col_offset in range(0, 3):
                    if col_center + col_offset != 0:
                        x_left.append(row)
                        y_left.append(col_offset + col_offset)
                        col_center = col_center + col_offset
                        break
                    if col_center - col_offset != 0:
                        x_left.append(row)
                        y_left.append(col_offset + col_offset)
                        col_center = col_center - col_offset
                        break

        for row in range(start_row, rows):
            curr_max_center_dist = start_center_dist + (max_center_dist - start_center_dist) * (row - start_row) // (
                        rows - start_row)
            for col_offset in range(0, curr_max_center_dist):
                # print(str(row) + ' ' + str(curr_max_center_dist) + ' ' + str(start_col - col_offset))
                if img[row][start_col + col_offset] != 0:
                    found_right = True
                    print("right found")
                    x_right.append(row)
                    y_right.append(start_col - col_offset)
            if found_right:
                break
        if not found_right:
            print("right not found")
            if found_left:
                x_right.append(0)
                x_right.append(rows)
                y_right.append(cols)
                y_right.append(cols)

        if not found_left:
            print("left not found")
            if found_right:
                x_left.append(0)
                x_left.append(rows)
                y_left.append(0)
                y_left.append(0)


        if found_right:
            i = 0
            for row in range(x_right[0] + 1, rows):
                col_center = y_right[i]
                for col_offset in range(0, 3):
                    if col_center + col_offset != 0:
                        x_right.append(row)
                        y_right.append(col_offset + col_offset)
                        col_center = col_center + col_offset
                        break
                    if col_center - col_offset != 0:
                        x_right.append(row)
                        y_right.append(col_offset + col_offset)
                        col_center = col_center - col_offset
                        break

        left.append((np.polyfit(x_left, y_left, 2)))
        right.append((np.polyfit(x_right, y_right, 2)))
    return left, right


def hough_lines(labels):
    for img in labels:
        print(img.shape)
        edges = cv2.Canny(img, 100, 255)
        threshold = 30
        min_line_length = 10
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, 0, min_line_length, 10)
        if not (lines is None or len(lines) == 0):
            # print lines
            for curr_line in lines:
                for line in curr_line:
                    # print line
                    cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)


def find_contours(labels):
    for img in labels:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 250, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print(len(contours))
        for contour in contours:
            color = (random.randrange(0, 256, 1), random.randrange(0, 256, 1), random.randrange(0, 256, 1))
            cv2.drawContours(img, contour, -1, color, 3)
        cv2.imshow('img', img)
        cv2.waitKey(0)

def main():
    labels = helper.read_images('../datas/images_roma/label/')  # , cv2.IMREAD_GRAYSCALE)
    size = (320, 256)
    labels = helper.resize_images(labels, size)
    # skeletons = create_skeleton(labels)
    # hough_lines(labels)
    find_contours(labels)

    # left, right = create_polynoms(skeletons)
    # print_labels(left, right, labels, (256, 320))


main()
