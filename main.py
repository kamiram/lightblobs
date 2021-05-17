import argparse
import cv2
import numpy as np


def main(filename, debug=False):
    print(f'process {filename}')
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    blur = image
    # blur = cv2.GaussianBlur(image, (5, 3), 1)

    ret, threshold = cv2.threshold(blur, 100, 255, 0)
    kernel = np.ones((10, 10), np.uint8)
    erode = cv2.erode(threshold, kernel, iterations=1)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 30

    detector = cv2.SimpleBlobDetector_create(params)
    blobs = detector.detect(image)

    print(f'lighten count: {len(blobs)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate light blocks')
    parser.add_argument('-f', '--filename', action='store', required=True, type=str, help='input filename')
    parser.add_argument('-d', '--debug', action='store_true', required=False, help='debug mode')
    args = parser.parse_args()

    main(args.filename, args.debug)
