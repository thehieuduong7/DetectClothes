#color detection
import cv2
import numpy as np
from B3_Predict import reg


def getContours(mask, img):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_canny = cv2.Canny(img_gray, 50, 50)
    contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    #print(hierachy)  #[Next, Previous, First_Child, Parent]
    #print(len(contours))
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area > 500):
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True),
                                      True)
            M = cv2.moments(approx)
            x, y, w, h = cv2.boundingRect(approx)
            img_sub = img[y:y + h, x:x + w]
            text = reg(img_sub)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            img = cv2.putText(img, text, (y, x), cv2.cv2.FONT_HERSHEY_COMPLEX,
                              1, (255, 0, 0), 2, cv2.LINE_AA)
    return img


img = cv2.imread('image/t_shirt_1.png')
img = cv2.resize(img, (700, 500))
update = False


def doSt(x):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 480)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 239, doSt)
cv2.createTrackbar("Hue Max", "TrackBars", 0, 255, doSt)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, doSt)
cv2.createTrackbar("Sat Max", "TrackBars", 0, 255, doSt)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, doSt)
cv2.createTrackbar("Val Max", "TrackBars", 0, 255, doSt)

while True:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    val_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    val_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    mask = cv2.inRange(img_hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    result = cv2.bitwise_and(img, img, mask=mask)
    result = getContours(mask, result)
    cv2.imshow("TrackBars", img)
    cv2.imshow("hsv", img_hsv)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()