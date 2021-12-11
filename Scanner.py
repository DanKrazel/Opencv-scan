import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys



def read(img):
    imgColor = cv2.imread(img, cv2.IMREAD_COLOR)
    return imgColor

def threshold(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return blackAndWhiteImage

def getContours(img):
    #get the num of the max contours
    (cnts, _) = cv2.findContours(threshold(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = cv2.contourArea(cnts[0])
    for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i])
        perimeter = cv2.arcLength(cnts[i], True)
        #print("Countour #", i, ": perimeter =", perimeter, " area =", area)
        if(area > maxArea):
            maxArea = area
            count = i
    return count


def findContours(img):
    # copy image
    imgCopy = img.copy()

    # find contours
    (cnts, _) = cv2.findContours(threshold(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = getContours(img)

    # get the approximation of the contours and angular points of the rectangle
    approx = cv2.approxPolyDP(cnts[count], 0.1 * cv2.arcLength(cnts[count], True), True)
    print(approx[0])

    # put 4 points on the rectangle
    imagePoints = cv2.drawContours(imgCopy, [approx], -1, (0, 255, 0), 25)
    #print(imagePoints[0])

    # get emplacement of the 4 points image
    pt_A = approx[0]
    pt_B = approx[1]
    pt_C = approx[2]
    pt_D = approx[3]
    print(pt_A, pt_B, pt_C, pt_D)

    # draw the contours
    cv2.drawContours(imagePoints, approx, -1, (255, 0, 0), 60)

    # get height and width of the rectangle
    (x, y, w, h) = cv2.boundingRect(cnts[count])
    print("width : {}".format(w))
    print("height : {}".format(h))

    # specify the input point and output point image
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                             [0, h - 1],
                             [w - 1, h - 1],
                             [w - 1, 0]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    #apply th computiong to the entire image
    out = cv2.warpPerspective(imgCopy, M, (w, h), flags=cv2.INTER_LINEAR)


    #imgContours = cv2.drawContours(imagePoints, cnts, count, (255, 0, 0), 25)
    return out



def main():
    input = cv2.imread(sys.argv[1])
    #plt.imshow(input)
    #plt.show()
    output = sys.argv[2]
    imgThreshold = threshold(input)
    imgContours = findContours(input)
    cv2.imwrite("images/output/imgThreshold.png", imgThreshold)

    cv2.imwrite("images/output/imgContours.png", imgContours)
    cv2.imwrite(output, imgContours)




if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
