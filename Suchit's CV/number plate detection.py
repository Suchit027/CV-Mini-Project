import cv2
import numpy as np
import imutils
from skimage.segmentation import clear_border


class PyImageSearchANPR:
    def __init__(self, minRatio=5, maxRatio=6, debug=False):
        self.minRatio = minRatio
        self.maxRatio = maxRatio
        self.debug = debug
        self.char_width = 438
        self.char_height = 260

    def debug_imshow(self, title, image, waitKey=True):
        if self.debug:
            cv2.imshow(title, image)
            if waitKey:
                cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):

        # performing blackhat morphological operation to get dark text on light regions
        # rectkernel defines the neighborhood for morphological operations
        # cv2.MORPH_RECT specifies that shape of kernel is rectangular
        # blackhat morphological operation highlight dark regions on light background
        # blackhat = closing - image; closing = dilation followed by erosion
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        self.debug_imshow('BLACKHAT', blackhat)

        # squareKernel is square neighborhood of pixels
        # closing morphological operation is used to fill small holes or gaps within bright region
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        thresh, light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.debug_imshow('LIGHT REGIONS', light)

        # Scharr operator is similar to Sobel operator but with better accuracy for first derivative
        # Scharr is better as it uses larger coefficients that highlight details more
        # [-3  0 3 ]
        # [-10 0 10]
        # [-3  0 3 ]
        # dx = 1 means calculate gradient in x direction; dy = 0 means don't calculate gradient in y direction
        # it is then followed by normalization operation
        gradX = cv2.Scharr(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype('uint8')
        self.debug_imshow('SCHARR GRADIENT', gradX)

        # smoothening is required to group the regions that may contain licence plate characters
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        _, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.debug_imshow('GRADIENT THRESHOLD', thresh)

        # to remove other large white regions
        thresh = cv2.erode(thresh, rectKernel, iterations=2)
        thresh = cv2.dilate(thresh, rectKernel, iterations=2)
        self.debug_imshow('GRADIENT ERODE/ DILATE', thresh)

        # the light image generated earlier is used to reveal the most appropriate candidates
        # more dilation and erosion for filling up the holes
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, rectKernel, iterations=2)
        thresh = cv2.erode(thresh, rectKernel, iterations=1)
        self.debug_imshow('FINAL', thresh, waitKey=True)

        # our license plate is not the largest but not the smallest either
        # imutils is used to handle different versions of opencv used
        # we are keeping only a specified number of contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[: keep]
        return contours

    def locate_license_plate(self, gray, candidates, clearBorder=False):
        plateContour = None
        region_interest = None
        for c in candidates:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if self.minRatio <= aspect_ratio <= self.maxRatio:
                plateContour = c
                plate = gray[y: y + h, x: x + w]
                _, region_interest = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                if clearBorder:
                    region_interest = clear_border(region_interest)
                self.debug_imshow('LICENSE PLATE', plate)
                self.debug_imshow('REGION OF INTEREST', region_interest, waitKey=True)
                break
        return region_interest, plateContour

    def find_and_ocr(self, image, clearBorder=False):
        plate_text = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        (plate, contour) = self.locate_license_plate(gray, candidates, clearBorder=clearBorder)
        if plate is not None:
            plate_copy = plate.copy()
            plate_copy = cv2.resize(plate_copy, (self.char_width, self.char_height), interpolation= cv2.INTER_LINEAR)
            self.debug_imshow('final plate image', plate_copy)
        return

ob = PyImageSearchANPR(debug= True)
img = cv2.imread('plate.jpeg', 1)
ob.find_and_ocr(img, clearBorder= True)
