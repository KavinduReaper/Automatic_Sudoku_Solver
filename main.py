import cv2 as cv


if __name__ == '__main__':
    capture = cv.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()
        print(ret)

        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    capture.release()
    cv.destroyAllWindows()
