import cv2
import numpy as np
import imutils
from skimage.segmentation import clear_border
from bisect import bisect_left


class PyImageSearchANPR:
    def __init__(self, minRatio=5, maxRatio=6, debug=False):
        self.minRatio = minRatio
        self.maxRatio = maxRatio
        self.debug = debug
        self.char_width = 438
        self.char_height = 260
        self.upper_chars_pos = []
        self.middle_chars_pos = []
        self.lower_chars_pos = []
        self.up_chars = 'ABCDEFGHIJKLM'
        self.mid_chars = 'NOPQRSTUVWXYZ'
        self.low_chars = '1234567890'

    def character_space_creation(self, image):
        image = cv2.GaussianBlur(image, (7, 7), 0)

        # finding negative of the given image
        def negative(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[: 2]
            for i in range(height):
                for j in range(width):
                    img[i][j] = 255 - img[i][j]
            return img

        nimg = negative(image)
        cv2.imwrite('char.png', nimg)
        return

    def debug_imshow(self, title, image, waitKey=True):
        if self.debug:
            cv2.imshow(title, image)
            if waitKey:
                cv2.waitKey(0)
            cv2.destroyAllWindows()

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

    def character_optimization(self):
        # img is character space
        char_space = cv2.imread('char.png', 1)
        height, width = char_space.shape[: 2]

        # resizing the char_space so that the template matches properly
        height = 3 * self.char_height
        width = 14 * (self.char_width // 10)
        char_space = cv2.resize(char_space, (width, height), interpolation=cv2.INTER_LINEAR)

        # saving the character space
        cv2.imwrite('char_space.png', char_space)
        self.debug_imshow('ALL CHARACTERS', char_space)

        # for defining character boundaries in the character space
        # it finds the center of those group of columns that have no white pixel
        # this allows us to store which character is placed where in the character space
        def character_space(image, store):
            i = 0
            while i < width:
                if np.sum(image[:, i]) == 0:
                    first, last = i, i
                    while i < width and np.sum(image[:, i]) == 0:
                        last = i
                        i += 1
                    store.append((last + first) // 2)
                else:
                    while np.sum(image[:, i]) > 0:
                        i += 1
            return

        # finding horizontal positions of characters in first half of character space
        first_half = char_space[: height // 3, :]
        self.debug_imshow('first half', first_half)
        character_space(first_half, self.upper_chars_pos)

        # finding horizontal positions of characters in middle of character space
        middle_half = char_space[height // 3: (2 * height) // 3, :]
        self.debug_imshow('middle half', middle_half)
        character_space(middle_half, self.middle_chars_pos)

        # creating letter character space
        cv2.imwrite('letter_char_space.png', char_space[: (2 * height) // 3])

        # creating number character space
        cv2.imwrite('number_char_space.png', char_space[(2 * height) // 3:])

        # finding horizontal positions of characters in second half of character space
        second_half = char_space[(2 * height) // 3:, :]
        self.debug_imshow('second half', second_half)
        character_space(second_half, self.lower_chars_pos)
        return

    def template_matching(self, template, track):
        # the letter from the plate that needs to be identified is called template
        height, width = template.shape[: 2]
        # character space containing all possible characters
        if track <= 2 or track == 5:
            space = cv2.imread('letter_char_space.png', 0)
        else:
            space = cv2.imread('number_char_space.png', 0)
        # result of template matching
        res = cv2.matchTemplate(space, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # storing top left corner of matched position of the template
        top_left = max_loc
        # storing bottom right corner of matched position of the template
        bottom_right = [width + top_left[0], height + top_left[1]]
        # creating a copy of character space to show which are was matched
        sc = space.copy()
        cv2.rectangle(sc, top_left, bottom_right, 125, 10)
        cv2.circle(sc, top_left, 50, 125, 10)
        cv2.circle(sc, bottom_right, 50, 125, 10)
        self.debug_imshow('a', sc)
        return (bottom_right[0] + top_left[0]) // 2, (top_left[1] + bottom_right[1]) // 2

    def template_creation(self, plate):
        # text of the license plate
        text = ''
        height, width = plate.shape[: 2]
        # i = current column in the image
        # track = number of blank spaces encountered up until that iteration; will help to figure out
        # whether the next character is letter or number
        i, track = 0, 0
        while i < width:
            # blank space
            if np.sum(plate[:, i]) == 0:
                while i < width and np.sum(plate[:, i]) == 0:
                    i += 1
                track += 1
            else:
                # start of character in the license plate
                region_left = i
                while i < width and np.sum(plate[:, i]) > 0:
                    i += 1
                # end of character in the license plate
                region_right = i
                # extracting the character
                template = plate[:, region_left: region_right + 1].copy()
                self.debug_imshow('a', template)

                pos = self.template_matching(template, track)

                # if the matched character belongs to lower characters
                if (track <= 2 or track == 5) and pos[1] >= self.char_height:
                    ind = bisect_left(self.middle_chars_pos, int(pos[0])) - 1
                    if ind >= len(self.mid_chars):
                        ind = len(self.mid_chars) - 1
                    text = text + self.mid_chars[ind]
                # if the matched characters belongs to upper characters
                elif (track <= 2 or track == 5) and pos[1] < self.char_height:
                    ind = bisect_left(self.upper_chars_pos, int(pos[0])) - 1
                    if ind >= len(self.up_chars):
                        ind = len(self.up_chars) - 1
                    text = text + self.up_chars[ind]
                else:
                    ind = bisect_left(self.lower_chars_pos, int(pos[0])) - 1
                    if ind >= len(self.low_chars):
                        ind = len(self.low_chars) - 1
                    text = text + self.low_chars[ind]

        return text

    def find_and_ocr(self, image, clearBorder=False):
        # text on license plate
        plate_text = ''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        (plate, contour) = self.locate_license_plate(gray, candidates, clearBorder=clearBorder)
        if plate is not None:
            plate_copy = plate.copy()

            # resizing to desired shape
            plate_copy = cv2.resize(plate_copy, (self.char_width, self.char_height), interpolation=cv2.INTER_LINEAR)

            # thresholding to separate the letters and to remove any dark patches in the white letter regions
            for i in range(self.char_height):
                for j in range(self.char_width):
                    if plate_copy[i][j] > 20:
                        plate_copy[i][j] = 255
                    else:
                        plate_copy[i][j] = 0

            # making the letters more thin
            rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 7))
            plate_copy = cv2.erode(plate_copy, rectKernel, iterations=1)
            self.debug_imshow('final plate image', plate_copy)

            plate_text = self.template_creation(plate_copy)

        return plate_text


if __name__ == '__main__':
    ob = PyImageSearchANPR(debug=True)
    # can choose c also
    char_space = cv2.imread('c1.png', 1)
    ob.character_space_creation(char_space)
    img = cv2.imread('plate.jpeg', 1)
    ob.character_optimization()
    license_plate = ob.find_and_ocr(img, clearBorder=True)
    print(license_plate)
