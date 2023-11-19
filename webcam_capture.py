import cv2

# Start the webcam
cap = cv2.VideoCapture(0)

# Define prev-frame
prev_frame = None

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform histogram equalization
    equ_frame = cv2.equalizeHist(gray_frame)

    cv2.imshow('equ', equ_frame)

    # Apply a blur filter
    blur_frame = cv2.GaussianBlur(equ_frame, (5, 5), 4)

    if prev_frame is None:
        prev_frame = blur_frame

    # Compute the absolute difference between the current frame and prev_frame
    diff = cv2.absdiff(prev_frame, blur_frame)
    cv2.imshow('diff', diff)

    # Compute the thresholded difference
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', thresh)

    # Compute the contours of the thresholded difference
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # Draw the contours on the frame
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    # cv2.imshow('contours', frame)

    # Draw the bounding boxes on the frame
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 40 and h > 40:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('contours', frame)

    # Update prev_frame
    prev_frame = blur_frame

    # Quit the program when 'q' is pressed
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
