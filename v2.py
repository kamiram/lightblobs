import argparse
import cv2
import numpy as np


def get_contours(image, min_bright):
    _, th1 = cv2.threshold(image, min_bright, 255, cv2.THRESH_TOZERO)
    thr = cv2.adaptiveThreshold(th1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    h, w = thr.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(thr, mask, (0, 0), 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filt_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= 70:
            filt_contours.append(contour)
    return filt_contours


def main(filename, level, debug=False):
    print(f'process {filename}')
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if debug:
        cv2.imwrite('debug-gray.png', image)

    image = cv2.addWeighted(image, 3, image, 0, 0.2)
    if debug:
        cv2.imwrite('debug-contrast.png', image)

    resized = cv2.resize(image, (800, 450), interpolation=cv2.INTER_AREA)
    if debug:
        cv2.imwrite('debug-resdized.png', image)

    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    if debug:
        cv2.imwrite('debug-blur.png', image)

    total_contours = get_contours(image_blur, 20)
    bright_contours = get_contours(image_blur, level)
    percent = len(bright_contours) / len(total_contours) * 100

    total_contours_img = resized
    bright_contours_img = resized.copy()
    cv2.drawContours(total_contours_img, total_contours, -1, (0, 255, 0), 1)
    cv2.drawContours(bright_contours_img, bright_contours, -1, (0, 0, 255), 1)

    if debug:
        cv2.imwrite('debug-total.png', image)
        cv2.imwrite('debug-good.png', image)

    print(f'Light %%: {percent:.1f} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate light blocks')
    parser.add_argument('-f', '--filename', action='store', required=True, type=str, help='input filename')
    parser.add_argument('-d', '--debug', action='store_true', required=False, help='debug mode')
    parser.add_argument('-l', '--level', action='store', type=int, required=False, default=127, help='light level')
    args = parser.parse_args()

    main(args.filename, args.level, args.debug)
