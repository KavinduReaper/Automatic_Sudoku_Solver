import operator
import os
import cv2 as cv
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def predict(image):
    model = load_model('model.h5')
    image = cv.resize(image, (28, 28)).astype('float32').reshape(1, 28, 28, 1)
    image /= 255
    prediction = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
    print(prediction.argmax())
    # print(prediction)
    return prediction.argmax()


def imageGrids(temp_Grid):
    # Creating the 9X9 grid of images
    Grid = []
    for i in range(0, len(temp_Grid) - 8, 9):
        Grid.append(temp_Grid[i:i + 9])
    # Converting all the cell images to np.array
    for i in range(9):
        for j in range(9):
            Grid[i][j] = np.array(Grid[i][j])
    try:
        for i in range(9):
            for j in range(9):
                os.remove("BoardCells/cell" + str(i) + str(j) + ".jpg")
    except:
        pass
    for i in range(9):
        for j in range(9):
            cv.imwrite(str("BoardCells/cell" + str(i) + str(j) + ".jpg"), Grid[i][j])
    return Grid


def predictDigits(Grid):
    tmp_Sudoku = [[0] * 9 for _ in range(9)]
    finalGrid = imageGrids(Grid)
    for i in range(9):
        for j in range(9):
            gray = cv.threshold(finalGrid[0][4], 128, 255, cv.THRESH_BINARY)[1]
            Contours = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            Contours = Contours[0] if len(Contours) == 2 else Contours[1]
            for c in Contours:
                x, y, w, h = cv.boundingRect(c)
                if w < 15 and h < 15:
                    tmp_Sudoku[i][j] = 0
                if x < 3 or y < 3 or h < 3 or w < 3:
                    # Note the number is always placed in the center
                    # Since image is 28x28
                    # the number will be in the center thus x >3 and y>3
                    # Additionally any of the external lines of the sudoku will not be thicker than 3
                    continue
                ROI = gray[y:y + h, x:x + w]
                # cv.imshow("Out1.png", ROI)
                # increasing the size of the number allows for better interpretation,
                # try adjusting the number and you will see the difference
                # ROI = scale_and_centre(ROI, 120)
                print(w, h)
                tmp_Sudoku[i][j] = predict(ROI)
                cv.imshow("1.png", ROI)
                cv.waitKey(0)
    return tmp_Sudoku


if __name__ == '__main__':
    img = cv.imread("./resources/fig2.PNG", cv.IMREAD_GRAYSCALE)
    proc = cv.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv.adaptiveThreshold(proc, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    proc = cv.bitwise_not(proc, proc)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    proc = cv.dilate(proc, kernel)

    # Find the corners of the largest contour
    contours, h = cv.findContours(proc, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    polygon = contours[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    crop_rect = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([distance_between(bottom_right, top_right),
                distance_between(top_left, bottom_left),
                distance_between(bottom_right, bottom_left),
                distance_between(top_left, top_right)])

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv.getPerspectiveTransform(src, dst)
    croppedImage = cv.warpPerspective(img.copy(), m, (int(side), int(side)))

    grid = cv.GaussianBlur(croppedImage, (9, 9), 0)
    # Adaptive thresholding the cropped grid and inverting it
    grid = cv.bitwise_not(cv.adaptiveThreshold(grid, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2))

    edge_h = np.shape(grid)[0]
    edge_w = np.shape(grid)[1]
    cellEdge_h = edge_h // 9
    cellEdge_w = np.shape(grid)[1] // 9

    tempGrid = []
    for i in range(cellEdge_h, edge_h + 1, cellEdge_h):
        for j in range(cellEdge_w, edge_w + 1, cellEdge_w):
            rows = grid[i - cellEdge_h:i]
            tempGrid.append([rows[k][j - cellEdge_w:j] for k in range(len(rows))])

    tmp_sudoku = predictDigits(tempGrid)

    # for ele in tmp_sudoku:
    #     print(ele)
    # cv.imshow("Out.png", grid)
    # cv.waitKey(0)
