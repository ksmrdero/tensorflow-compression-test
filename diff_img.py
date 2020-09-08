from skimage.measure import compare_ssim
import cv2
import numpy as np
import os
# before = cv2.imread('left.png')
# after = cv2.imread('right.png')


def write_diff(before, after):
    im1 = before.copy()
    im2 = after.copy()
    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = compare_ssim(before_gray, after_gray, full=True)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)
    diff = np.repeat(diff[:, :, np.newaxis], 3, axis=2)
    return np.concatenate((im1, im2, diff, filled_after), axis=1), score


def main():
    DIRECTORY = 'results'
    path = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    sim = open('similarities.txt', 'w')
    files = os.listdir(path)
    count = 0
    for idx, file in enumerate(files):

        if file.split('.')[-1] == 'png' and 'tfci' not in file:
            count += 1
            print('Processing', count)
            reconstructed = file + '.tfci.png'
            im1 = cv2.imread(file)
            im2 = cv2.imread(reconstructed)
            
            if not os.path.isfile(reconstructed):
                continue
            res, score = write_diff(im1, im2)
            # res.save(path + '/' + DIRECTORY + '/' + file)
            res = cv2.resize(res, (0, 0), fx=0.5, fy=0.5)
            cv2.imwrite(path + '/' + DIRECTORY + '/' + file, res)
            sim.write('{} Score: {:.4f}\n'.format(file, score))
    sim.close()


if __name__ == "__main__":
    main()
