import sys
import os
import numpy as np
import cv2
import imutils

# Source: https://github.com/niconielsen32/ComputerVision/tree/master/imageStitching

def main(source_num):
    directory = f"./data/source{source_num}"
    # image_paths = [os.path.join(directory, file) for file in os.listdir(directory)]
    image_paths = [os.path.join(directory, f'{source_file}') for source_file in sorted(os.listdir(directory))]
    images = []

    for image in image_paths[::-1]:
        img = cv2.imread(image)
        images.append(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1000)

    imageStitcher = cv2.Stitcher_create()

    error, stitched_img = imageStitcher.stitch(images)

    if not error:
        output_path = f"./output/{source_num}_stitchedOutput.png"
        cv2.imwrite(output_path, stitched_img)
        cv2.imshow("Stitched Img", stitched_img)
        cv2.waitKey(2000)

        stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

        gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow("Threshold Image", thresh_img)
        cv2.waitKey(1000)

        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)

        mask = np.zeros(thresh_img.shape, dtype="uint8")
        x, y, w, h = cv2.boundingRect(areaOI)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        minRectangle = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            minRectangle = cv2.erode(minRectangle, None)
            sub = cv2.subtract(minRectangle, thresh_img)

        contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)

        cv2.imshow("minRectangle Image", minRectangle)
        cv2.waitKey(1000)

        x, y, w, h = cv2.boundingRect(areaOI)

        stitched_img = stitched_img[y:y + h, x:x + w]

        output_path = f"./output/{source_num}_processedOutput.png"
        cv2.imwrite(output_path, stitched_img)

        cv2.imshow("Stitched Image Processed", stitched_img)

        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    else:
        print("Images could not be stitched!")
        if error == 1:
            print(f"Error {error}: Likely not enough keypoints being detected!")
        elif error == 2:
            print(f"Error {error}: RANSAC homography estimation failed!")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <source numbers in ./data/>")
        sys.exit(1)

    for i in range(1, len(sys.argv)):
        source_num = sys.argv[i]
        main(source_num)
