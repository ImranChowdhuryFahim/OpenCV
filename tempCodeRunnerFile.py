
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
                mouse.scroll(0, 1)
            elif count_defects == 3:
                check = 0
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
            while mouse.position!=(sx-int((mouseLoc[0]*sx)/camx),int((mouseLoc[1]*sy)/camy)):
                pass

            #setting the old location to the current mouse location
            mLocOld=mouseLoc
            #print(mouseLoc)
            #these variables were added so that we get the outer rectangle that combines both objects 
            openx,openy,openw,openh=cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+h2]]]))
            #print(check)
        #when there's only when object detected it will act as a left click mouse    
        elif(len(conts)==1 and check == 1):
            x,y,w,h=cv2.boundingRect(conts[0])

            # we check first and we allow the press fct if it's not pressed yet
            #we did that to avoid the continues pressing 
            if(isPressed==0):

                if(abs((w*h-openw*openh)*100/(w*h))<30): #the difference between th combined rectangle for both objct and the 
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
                while mouse.position!=(sx-int((mouseLoc[0]*sx)/camx),int((mouseLoc[1]*sy)/camy)):
                    pass
                mLocOld=mouseLoc
        elif (len(conts) > 3):
            print ( " Remove yellow color from the background ")
            #break

            
            
        #showing the results 
        cv2.imshow("Virtual mouse",img)

#Ending clicking and double click

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory

camera.release()
cv2.destroyAllWindows()
        