import numpy as np
import cv2
import math
from PIL import ImageGrab
import time
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data


prevRoi = [[0,0],[1,0],[1,1],[0,1]]
clockCenter = (0,0)
glob_clockhands = [[], []]
glob_lineCoords = [(), ()];

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            return cv2.line(img,(coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
    except:
        pass
           

#Get region of interest given an image and vertices
def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def auto_canny(image, sigma=0.33):
    #compute median
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def process_img(original_image):
    global prevRoi
    global clockCenter
    global glob_lineCoords
    global glob_clockhands
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    output = original_image.copy()
    # detect circles in the image
    #circles = cv2.HoughCircles(processed_img, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = cv2.HoughCircles(processed_img, cv2.HOUGH_GRADIENT, 1.1, 100, maxRadius=300)

    roiDim = [] 
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
     
        # loop over the (x, y) coordinates and radius of the circles
        maxR = 0
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            clockCenter=(x,y)
            if(r > maxR):
                roiDim = [[x-r-5, y-r-5], [x+r+5, y-r-5], [x+r+5, y+r+5],[x-r-5, y+r+5]]
                prevRoi = roiDim
    
    if len(roiDim) == 0:
        roiDim = prevRoi

    #kernel = np.ones((3,3),np.uint8)
    #processed_img = cv2.erode(processed_img,kernel,iterations = 1)

    #processed_img = auto_canny(processed_img)
    #see ROI notes in notebook
    vertices = np.array(roiDim)
    processed_img = roi(processed_img, [vertices])
    
    #processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)
    #edges = cv2.Canny(processed_img,50, 150, apertureSize=3)

    kernel = np.ones((6,6),np.uint8)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    minLineLength = 50
    maxLineGap = 5



    #each hand has a x0,x1,y0,y1 and an angle where 0 is hours, 1 is minutes, 2 is seconds

    edges = cv2.Canny(processed_img, 50, 200)
    #edges = auto_canny(processed_img)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    lines = probabilistic_hough_line(edges, threshold=5, line_length=minLineLength, line_gap=maxLineGap)
    #lines = cv2.HoughLines(edges, 1, np.pi/180, 60)
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength, maxLineGap)

    #clockhands = [[(0,0), (1,1), np.pi], [(0,0), (q1,1), np.pi], [(0,0), (1,1), np.pi]]
    #maxLine = 0
    #minLine = 1000
    lineCoords = [[]]
    x, y = clockCenter
    lineAngs = []
    newAng = True
    maxima1 = 0
    maxima2 = 0

    distCenter = 15

    clockhands = [0, 0]
    if lines is not None:
        for line in lines:
            # for x1,y1,x2,y2 in line:
            p0, p1 = line
            x1, y1 = p0
            x2, y2 = p1

            lenLine = ((x2-x1) ** 2 + (y2-y1) ** 2) ** 0.5

            if(abs(x-x1) < distCenter and abs(y-y1) < distCenter and lenLine > minLineLength):
                lineCoords.append([(x1,y1), (x2,y2)])
                
                #ang = np.arctan2((y1-y2),(x2-x1))

                #Bottom of screen = max(y), rotate unit circle to match the clock
                ang = np.arctan2((x2-x1),(y1-y2))
                ang = ang * 180 / math.pi
                ang = (ang + 360) % 360

                for lineAng in lineAngs:
                    if(abs(abs(ang)- abs(lineAng)) < 5):
                        newAng = False      #Keep False

                if(lenLine > maxima1 and newAng):
                    maxima1 = lenLine
                    clockhands[0] = ang
                elif(lenLine > maxima2 and newAng):
                    maxima2 = lenLine
                    clockhands[1] = ang


                if(newAng):
                    lineAngs.append(ang)                           
                    cv2.line(original_image,(x1,y1),(x2,y2),(0,0,255),2)

                if(len(glob_clockhands[0]) == 0):
                    glob_clockhands[0] = [ang, lenLine]
                    glob_lineCoords[0] = [(x1,y1), (x2,y2)]
                elif(len(glob_clockhands[1]) == 0):
                    if(abs(ang - glob_clockhands[0][0]) > 10):
                        glob_clockhands[1] = [ang, lenLine]
                        glob_lineCoords[1] = [(x1,y1), (x2,y2)]
                    # else:
                        # for i in range(0, len(glob_clockhands)):
                        #     if(abs(abs(glob_clockhands[i][0]) - abs(ang))>5)


            # for rho,theta in line:
            #     a = np.cos(theta)
            #     b = np.sin(theta)

            #     x0 = a*rho
            #     y0 = b*rho
            #     x1 = int(x0 + 1000*(-b))
            #     y1 = int(y0 + 1000*(a))
            #     x2 = int(x0 - 1000*(-b))
            #     y2 = int(y0 - 1000*(a))

            #     lenLine = ((x2-x1) ** 2 + (y2-y1) ** 2) ** 0.5
            #     if(abs(x-x1) < 25 and abs(y-y1) < 25):
            #         cv2.line(processed_img,(x1,y1),(x2,y2),(0,255,0),2)

    # clockhands[0] = clockhands[0] - 90
    # clockhands[1] = clockhands[1] - 90


    #print(clockhands)


    #kernel = np.ones((5,5),np.uint8)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel )

    #lines = cv2.HoughLinesP(edges, 1, np.pi/180,10,100, minLineLength, maxLineGap)
    #newIm = draw_lines(np.zeros(processed_img.shape), lines)
    return original_image

def computeTime(ang_H, ang_M, ang_S=0):
    mm = round(ang_M / 6)
    ss = round(ang_S / 6)

    errHH = round((ang_H / 6) /5)
    HH = (ang_H / 6) // 5

    if(ss == 60):
        mm += 1
    if(mm == 60):
        HH += 1
    elif(mm < 30 and (errHH != HH)):
        HH += 1
    if(HH == 0):
        HH = 12

    return [int(HH), int(mm), int(ss)]

def timeToString(temp):
    for i in range(0, len(temp)):
        temp[i] = str(temp[i])

    return ":".join(temp)

def main():
    last_time = time.time()
    count = 0
    while(count < 20):
        screen = ImageGrab.grab(bbox=(0, 100, 750, 600)) #x, y, w , h)
        #screen = cv2.imread('clock4.jpg')

        #screen_np= cv2.resize(np.array(screen), (960,540))
        screen_np = np.array(screen)
        new_screen = process_img(screen_np)
        #print('Loop took {} seconds'.format(time.time() -last_time))
        last_time = time.time()
        try:
            cv2.imshow('window', new_screen)
        except:
            print("Imshow error")
        count += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    screen = ImageGrab.grab(bbox=(0, 100, 750, 600)) #x, y, w , h)
    screen_np = np.array(screen)
    print(glob_clockhands)
    for line in glob_lineCoords:
        cv2.line(screen_np,line[0],line[1],(0,0,255),2)
    try:
        cv2.imshow('window', screen_np)
    except:
        print("Imshow error")

    ang_H = 0
    ang_M = 0
    if(glob_clockhands[0][1] > glob_clockhands[1][1]):
        ang_H = glob_clockhands[1][0]
        ang_M = glob_clockhands[0][0]
    else:
        ang_H = glob_clockhands[0][0]
        ang_M = glob_clockhands[1][0]

    clocktime = computeTime(ang_H, ang_M)
    print(timeToString(clocktime))

    cv2.waitKey()
    cv2.destroyAllWindows()
main()