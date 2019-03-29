import numpy as np
import cv2
from PIL import ImageGrab
import time

prevRoi = [[0,0],[1,0],[1,1],[0,1]]
clockCenter = (0,0)

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
    edges = cv2.Canny(processed_img,50, 150, apertureSize=3)

    minLineLength = 30
    maxLineGap = 5

    #lines = cv2.HoughLines(edges, 1, np.pi/180, 60)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength, maxLineGap)
    #each hand has a x0,x1,y0,y1 and an angle where 0 is hours, 1 is minutes, 2 is seconds
    clockhands = [[(0,0), (1,1), np.pi], [(0,0), (1,1), np.pi], [(0,0), (1,1), np.pi]]
    maxLine = 0
    minLine = 1000
    # x, y = clockCenter
    # if lines is not None:
    #     for line in lines:
    #         for x1,y1,x2,y2 in line:
    #             cv2.line(processed_img,(x1,y1),(x,y),(0,0,255),2)
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

            #     cv2.line(original_image,(x1,y1),(x2,y2),(0,255,0),2)

    #kernel = np.ones((5,5),np.uint8)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel )

    #lines = cv2.HoughLinesP(edges, 1, np.pi/180,10,100, minLineLength, maxLineGap)
    #newIm = draw_lines(np.zeros(processed_img.shape), lines)

    return processed_img

def main():
    last_time = time.time()
    while(True):
        screen = ImageGrab.grab(bbox=(0, 100, 750, 600)) #x, y, w , h)
        
        #screen_np= cv2.resize(np.array(screen), (960,540))
        screen_np = np.array(screen)
        new_screen = process_img(screen_np)
        #print('Loop took {} seconds'.format(time.time() -last_time))
        last_time = time.time()
        try:
            cv2.imshow('window', new_screen)
        except:
            print("Imshow error")
            
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
main()