# Importing libraries
import numpy as np 
import cv2 

# Get video
cam = cv2.VideoCapture(0)

while(1):

    # Read video frames
    _, image = cam.read()

    # Converting BGR to HSV
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

    # Setting range and defining mask for colors
    red_lower = np.array([136, 87, 111], np.uint8) 
    red_upper = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

    green_lower = np.array([25*1.5, 52*1.5, 72*1.5], np.uint8) 
    green_upper = np.array([102, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

    blue_lower = np.array([94, 80, 2], np.uint8) 
    blue_upper = np.array([120, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

    orange_lower = np.array([10, 110, 110], np.uint8) 
    orange_upper = np.array([25, 255, 255], np.uint8) 
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    # Defining kernel size for dilation 
    kernel = np.ones((5, 5), "uint8")

    # Making objects bold
    red_mask = cv2.dilate(red_mask, kernel) 
    green_mask = cv2.dilate(green_mask, kernel) 
    blue_mask = cv2.dilate(blue_mask, kernel) 
    orange_mask = cv2.dilate(orange_mask, kernel) 

    # Draw rectangles for colors 
    red_count = 0
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 220): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2) 
            cv2.putText(image, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            red_count = red_count + 1	 

    green_count = 0
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 220): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            cv2.putText(image, "Green", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0)) 
            green_count = green_count + 1

    blue_count = 0
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 220): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            cv2.putText(image, "Blue", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
            blue_count = blue_count + 1 

    orange_count = 0
    contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 220): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2) 
            cv2.putText(image, "Orange", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255))
            orange_count = orange_count + 1 

    w, h, _ = image.shape
    x, y = 25, 25
    cv2.putText(image, f"Red: {red_count}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
    cv2.putText(image, f"Green: {green_count}", (x, 2*y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
    cv2.putText(image, f"Blue: {blue_count}", (x, 3*y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
    cv2.putText(image, f"Orange: {orange_count}", (x, 4*y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255))
                
    cv2.imshow("Detected colors", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows() 
        break
