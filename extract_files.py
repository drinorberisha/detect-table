from typing import List
import os, glob, cv2
import matplotlib.pyplot as plt
import numpy as np


class BoxAnnotation():
    def __init__(self, x: float, y: float, width: float, height: float, class_name: str):
        # this is a common data structure we use in many projects, pls use it as it is.
        self.x = x  # relative per image width
        self.y = y  # relative per image height
        self.width = width  # relative per image width
        self.height = height  # relative per image height
        self.class_name = class_name


class TableAnalysis:
    config = {
        "show_plots": False,
    }

    @staticmethod
    def sort_contours(cnts, method="top-to-bottom"):
        reverse = False
        i = 0

        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        bounding_boxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return cnts, bounding_boxes

    @staticmethod
    def create_boxes_from_contours(contours):
        boxes = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 1000 and h < 500:
                boxes.append([x, y, w, h])

        return boxes

    def get_vertical_horizontal_lines_from_image(self, img):
        thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = 255 - img_bin

        kernel_len = np.array(img).shape[1] // 100
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

        image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

        if self.config["show_plots"]:
            plotting = plt.imshow(image_1, cmap='gray')
            plt.show()

        image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

        if self.config["show_plots"]:
            plotting = plt.imshow(image_2, cmap='gray')
            plt.show()

        return vertical_lines, horizontal_lines

    def combine_horizontal_vertical_lines(self, vertical_lines, horizontal_lines, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

        img_vh = cv2.erode(~img_vh, kernel, iterations=2)
        thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        bitxor = cv2.bitwise_xor(img, img_vh)
        bitnot = cv2.bitwise_not(bitxor)

        if self.config["show_plots"]:
            plotting = plt.imshow(bitnot, cmap='gray')
            plt.show()

        return img_vh

    @staticmethod
    def store_boxes_to_column_row(box, mean):
        row = []
        column = []

        for i in range(len(box)):
            if i == 0:
                column.append(box[i])
                previous = box[i]
            else:
                if box[i][1] <= previous[1] + mean / 2:
                    column.append(box[i])
                    previous = box[i]
                    if i == len(box) - 1:
                        row.append(column)
                else:
                    row.append(column)
                    column = []
                    previous = box[i]
                    column.append(box[i])

        return row, column

    @staticmethod
    def create_final_boxes(row, column):
        countcol = 0
        for i in range(len(row)):
            countcol = len(row[i])
            if countcol > countcol:
                countcol = countcol

        center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
        center = np.array(center)
        center.sort()

        finalboxes = []
        for i in range(len(row)):
            lis = []
            for k in range(countcol):
                lis.append([])
            for j in range(len(row[i])):
                diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
                minimum = min(diff)
                indexing = list(diff).index(minimum)
                lis[indexing].append(row[i][j])
            finalboxes.append(lis)

        return finalboxes

    def process(self, filepath) -> List[BoxAnnotation]:
        box_annotations = []

        img = cv2.imread(filepath, 0)
        image_height, image_width = img.shape

        vertical_lines, horizontal_lines = self.get_vertical_horizontal_lines_from_image(img)
        img_vh = self.combine_horizontal_vertical_lines(vertical_lines, horizontal_lines, img)

        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        box_annotations.append(BoxAnnotation(x=0, y=0, width=1, height=1, class_name="cell-all"))
        contours, boundingBoxes = self.sort_contours(contours, method="default")

        # Creating a list of heights for all detected boxes
        heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
        mean = np.mean(heights)

        box = self.create_boxes_from_contours(contours)
        row, column = self.store_boxes_to_column_row(box, mean)
        final_boxes = self.create_final_boxes(row, column)

        for col_num, col_box in enumerate(final_boxes):
            for boxes in col_box:
                for row_num, box in enumerate(boxes):
                    if box[3] / image_height > 0.005 and box[2] / image_width > 0.005:
                        box_annotations.append(
                            BoxAnnotation(x=box[0] / image_width, y=box[1] / image_height, width=box[2] / image_width,
                                          height=box[3] / image_height, class_name="cell-{}-{}".format(col_num, row_num)))

        return box_annotations

    def write_results(self, box_annotations: List[BoxAnnotation], filepath, outdir):
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        image_height, image_width, num_channels = image.shape

        for annotation in box_annotations:
            x1 = max(0, int(annotation.x * image_width))
            y1 = max(0, int(annotation.y * image_height))
            x2 = min(image_width - 1, x1 + int(annotation.width * image_width))
            y2 = min(image_height - 1, y1 + int(annotation.height * image_height))

            cell = image[y1:y2, x1:x2, ...]
            os.makedirs(outdir, exist_ok=True)
            cv2.imwrite(os.path.join(outdir, annotation.class_name + ".png"), cell)


if __name__ == "__main__":
    # this part calls the process function above for every picture (pls. dont change it)
    table_analysis = TableAnalysis()

    for filepath in glob.glob(os.path.join("tables/*.png")):
        print("Executing for:", filepath)

        box_annotations = table_analysis.process(filepath)
        outdir = os.path.join("results", os.path.basename(filepath).split(".")[0])
        table_analysis.write_results(box_annotations, filepath, outdir)
