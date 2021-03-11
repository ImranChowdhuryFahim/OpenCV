import cv2
import sys
import numpy as np
import imutils
import math
import wx
from pynput.mouse import Button ,Controller
def nothing(x):
    pass

useCamera=False

# Check if filename is passed
if (len(sys.argv) <= 1) :
    print("'Usage: python hsvThresholder.py <ImageFilePath>' to ignore camera and use a local image.")
    useCamera = True

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX MIN HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 30)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

cv2.setTrackbarPos('HMin', 'image', 10)
cv2.setTrackbarPos('SMin', 'image', 130)
cv2.setTrackbarPos('VMin', 'image', 80)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# Output Image to display
if useCamera:
    cap = cv2.VideoCapture(0)
    # Wait longer to prevent freeze for videos.
    waitTime = 330
else:
    img = cv2.imread(sys.argv[1])
    output = img
    waitTime = 33

while(1):

    if useCamera:
        # Capture frame-by-frame
        ret, img = cap.read()
        output = img

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        #print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('image',output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(waitTime) & 0xFF == ord('y'):
        break

# Release resources
if useCamera:
    cap.release()
cv2.destroyAllWindows()
# organize imports
print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        
needscroll = False 
mouse = Controller()
#https://pythonhosted.org/pynput/mouse.html

# global variables
bg = None

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


    #---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)



#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    cap = camera
    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590
 #click and double click   
    app=wx.App(False)
    (sx,sy)=wx.GetDisplaySize()
    (camx,camy)=(320,240)
    cap.set(3,camx)
    cap.set(4,camy)

    # range for HSV (green color)
    # lower_g=np.array([33,70,30])
    # upper_g=np.array([102,255,255])
    # lower_g=np.array([110,70,30])
    # upper_g=np.array([102,255,255])
    # lower_g = np.array([155,25,0])
    # upper_g = np.array([179,255,255])
    lower_g = np.array([phMin,psMin,pvMin])
    upper_g = np.array([phMax, psMax, pvMax])

    #Kerenel
    kernelOpen=np.ones((5,5))
    kernelClose=np.ones((20,20))

    mLocOld=np.array([0,0])
    mouseLoc=np.array([0,0])

    DampingFactor=2 #Damping factor must be greater than 1

    isPressed=0
    openx,openy,openw,openh=(0,0,0,0)
    # initialize num of frames
    num_frames = 0
#  clicking and double click

    # keep looping, until interrupted
    while(True):
        #checking the state of mouse
        check = 1
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=600)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1
        try:
            # Find contour with maximum area
            contour = segmented
            crop_image = roi

            # Create bounding rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

            # Find convex hull
            hull = cv2.convexHull(contour)
            areacnt = cv2.contourArea(contour)

            # Draw contour
            drawing = np.zeros(crop_image.shape, np.uint8)
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
            cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

            # Find convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)

            # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
            # tips) for all defects
            count_defects = 0
            count = 0

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                # if angle > 90 draw a circle at the far point
                if angle <= 80:
                    count_defects += 1
                    cv2.circle(crop_image, far, 3, [0, 0, 255], -1)

                cv2.line(crop_image, start, end, [0, 255, 0], 2)

            # Print number of fingers   
            
            # if count_defects == 0:
            #     print(areacnt)
            #     if areacnt > 3050 :
            #         cv2.putText(frame, "yes", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            #     else:
            #         cv2.putText(frame, "no", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)

            if count_defects == 4:
                check = 0
                #print(check)
                mouse.scroll(0, 1)
            elif count_defects == 3:
                check = 0
                #print(check)
                mouse.scroll(0,-1)
            elif count_defects < 3:
                check = 1

                
            else:
                pass
        except:
            pass
        # display the frame with segmented hand
        #cv2.imshow("Video Feed", clone)
        # Show required images
        # cv2.imshow("Gesture", frame)
        #all_image = np.hstack((drawing, crop_image))
        #cv2.imshow('Contours', all_image)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

# clicking and double click

        ret,img=cap.read()

        img=cv2.resize(img,(440,220))
        imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(imgHSV,lower_g,upper_g)

        #using morphology to erase noise as maximum as possible 
        new_mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
        another_mask=cv2.morphologyEx(new_mask,cv2.MORPH_CLOSE,kernelClose)
        final_mask=another_mask
        
        conts,h=cv2.findContours(final_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #print("after: ",check)
        # Once 2 objects are detected the center of there distance will be the reference on controlling the mouse
        if(len(conts)==2 and check == 1):

            #if the button is pressed we need to release it first
            if(isPressed==1):
                isPressed=0
                mouse.release(Button.left)

            #drawing the rectagle around both objects
            x1,y1,w1,h1=cv2.boundingRect(conts[0])
            x2,y2,w2,h2=cv2.boundingRect(conts[1])
            cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
            cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)

            #the line between the center of the previous rectangles
            cx1=int(x1+w1/2)
            cy1=int(y1+h1/2)
            cx2=int(x2+w2/2)
            cy2=int(y2+h2/2)
            cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),2)

            #the center of that line (reference point)
            clx=int((cx1+cx2)/2)
            cly=int((cy1+cy2)/2)
            cv2.circle(img,(clx,cly),2,(0,0,255),2)

            #adding the damping factor so that the movement of the mouse is smoother
            mouseLoc=mLocOld+((clx,cly)-mLocOld)/DampingFactor
            mouse.position=(sx-int((mouseLoc[0]*sx)/camx),int((mouseLoc[1]*sy)/camy))
            # print(mouse.position[0])
            # print(mouseLoc)
            # while mouse.position!=(sx-int((mouseLoc[0]*sx)/camx),int((mouseLoc[1]*sy)/camy)) or mouse.position[0] == 0:
            #     pass

            #setting the old location to the current mouse location
            mLocOld=mouseLoc
            #print(mouseLoc)
            #these variables were added so that we get the outer rectangle that combines both objects 
            openx,openy,openw,openh=cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+h2]]]))
            #print(check)
        #when there's only when object detected it will act as a left click mouse    
        elif(len(conts)==1):
            x,y,w,h=cv2.boundingRect(conts[0])
            #print(len(conts))
            # we check first and we allow the press fct if it's not pressed yet
            #we did that to avoid the continues pressing 
            if(isPressed==0):
                print(abs((w*h-openw*openh)*100/(w*h)))
                if(abs((w*h-openw*openh)*100/(w*h))<230): #the difference between th combined rectangle for both objct and the 
                    isPressed=1                          #the outer rectangle is not more than 30%
                    mouse.press(Button.left)
                    openx,openy,openw,openh=(0,0,0,0)

            #this else was added so that if there's only one object detected it will not act as a mouse  
            else:
                #getting rectangle coordinates and drawing it 
                x,y,w,h=cv2.boundingRect(conts[0])
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

                #getting the center of the circle that will be inside the outer rectangle
                cx=int(x+w/2)
                cy=int(y+h/2)
                cv2.circle(img,(cx,cy),int((w+h)/4),(0,0,255),2)#drawing that circle

                mouseLoc=mLocOld+((cx,cy)-mLocOld)/DampingFactor
                mouse.position=(sx-int((mouseLoc[0]*sx)/camx),int((mouseLoc[1]*sy)/camy))
                while mouse.position!=(sx-int((mouseLoc[0]*sx)/camx),int((mouseLoc[1]*sy)/camy)) or mouse.position[0] == 0:
                    pass
                mLocOld=mouseLoc
        elif (len(conts) > 3):
            print ( " Remove yellow color from the background ")
            break

            
            
        #showing the results 
        cv2.imshow("Virtual mouse",img)

#Ending clicking and double click

        # if the user pressed "q", then stop looping
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break

# free up memory

camera.release()
cv2.destroyAllWindows()
        
