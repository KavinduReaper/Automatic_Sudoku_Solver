# import the necessary packages
import cv2
import imutils
import numpy as np
import pytesseract
import tensorflow as tf
from sudoku import Sudoku
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from pyimagesearch.sudoku import extract_digit
from pyimagesearch.sudoku import find_puzzle

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

    # load the digit classifier from disk
    print("[INFO] loading digit classifier...")
    model = load_model("Model/train_digit_classifier.h5")
    # train_digit_classifier
    # digitClassifierModel

    # load the Input image from disk and resize it
    print("[INFO] processing image...")
    image = cv2.imread("resources/fig2.PNG")
    image = imutils.resize(image, width=600)

    # find the puzzle in the image and then
    # if debug = true, it shows the image
    (puzzleImage, warped) = find_puzzle(image, debug=True)

    # initialize our 9x9 sudoku board
    board = np.zeros((9, 9), dtype="int")

    # a sudoku puzzle is a 9x9 grid (81 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cellLocs = []

    # loop over the grid locations
    for y in range(0, 9):
        # initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            # compute the starting and ending (x, y)-coordinates of the current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))

            # crop the cell from the warped transform image and then extract the digit from the cell
            cell = warped[startY:endY, startX:endX]
            # if debug = true, it shows the image of every digits
            digit = extract_digit(cell, debug=False)

            # verify that the digit is not empty
            if digit is not None:
                foo = np.hstack([cell, digit])
                # cv2.imshow("Cell/Digit", foo)

                # resize the cell to 28x28 pixels and then prepare the
                # cell for classification
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)

                roi = np.expand_dims(roi, axis=0)
                # classify the digit and update the sudoku board with the
                # prediction
                pred = model.predict(roi).argmax(axis=1)[0]
                print(f"Predicted Number: {pred}")
                # num = pytesseract.image_to_string(roi, config=r'--oem 3 --psm 6 outputbase digits')
                # print("num : ", num)

                board[y, x] = pred
            else:
                board[y, x] = 0

        # add the row to our cell locations
        cellLocs.append(row)

    # construct a sudoku puzzle from the board
    print("[INFO] OCR'd sudoku board:")
    puzzle = Sudoku(3, 3, board=board.tolist())
    puzzle.show()

    # solve the sudoku puzzle
    print("[INFO] solving sudoku puzzle...")
    solution = puzzle.solve()
    solution.show_full()

    # print("[INFO] Solving sudoku puzzle using defined algorithm")
    # Input = puzzle.board
    # for r in range(len(Input)):
    #     for c in range(len(Input[0])):
    #         if Input[r][c] is None:
    #             Input[r][c] = 0
    #     print(Input[r])
    # solution = solveSudoku(Input, all_solutions=False)

    # loop over the cell locations and board
    for (cellRow, boardRow) in zip(cellLocs, solution.board):
        # loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # unpack the cell coordinates
            startX, startY, endX, endY = box

            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY

            # draw the result digit on the sudoku puzzle image
            cv2.putText(puzzleImage, str(digit), (textX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Sudoku Result", puzzleImage)
    cv2.waitKey(0)
