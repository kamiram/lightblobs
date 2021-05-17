import argparse
import cv2
import numpy as np


def main(filename, level=127, debug=False):
    print(f'process {filename}')
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if debug:
        cv2.imwrite('debug-gray.png', image)

    image = cv2.addWeighted(image, 3, image, 0, 0.2)
    if debug:
        cv2.imwrite('debug-contrast.png', image)

    ret, threshold = cv2.threshold(image, level, 255, 0)
    if debug:
        cv2.imwrite('debug-threshold.png', threshold)

    kernel = np.ones((20, 20), np.uint8)
    erode = cv2.erode(threshold, kernel, iterations=1)
    if debug:
        cv2.imwrite('debug-erode.png', threshold)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 50

    detector = cv2.SimpleBlobDetector_create(params)
    blobs = detector.detect(threshold)

    print(f'lighten count: {len(blobs)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate light blocks')
    parser.add_argument('-f', '--filename', action='store', required=True, type=str, help='input filename')
    parser.add_argument('-d', '--debug', action='store_true', required=False, help='debug mode')
    parser.add_argument('-l', '--level', action='store', type=int, required=False, default=127, help='light level')
    args = parser.parse_args()

    main(args.filename, args.level, args.debug)
