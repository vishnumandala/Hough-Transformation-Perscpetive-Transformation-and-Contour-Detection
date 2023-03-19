import cv2
import numpy as np
import matplotlib.pyplot as plt

'''------------------------------------Problem 2------------------------------------------------------------'''
cap = cv2.VideoCapture('ball.mov')  # Read the video file

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # Convert the frame to HSV color space
    mask = cv2.inRange(hsv, (0,150,130), (6,255,255))   # Filter the Red Channel using color thresholding

    # Find the center point of the ball using color thresholding
    ycoords, xcoords = np.where(mask > 0)
    if len(ycoords) > 0 and len(xcoords) > 0:   # Check if the ball is in the frame 
        center = (int(np.mean(xcoords)), int(np.mean(ycoords)))
    else:
        center = None

    if center is not None:
        # Define the search region around the known center
        search_region = frame[max(center[1]-50, 0):min(center[1]+50, frame.shape[0]), max(center[0]-50, 0):min(center[0]+50, frame.shape[1])]   

        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)  # Convert the search region to grayscale    
        blur = cv2.GaussianBlur(gray, (5, 5), 0)    # Apply Gaussian blur to reduce noise
        edges = cv2.Canny(blur, 50, 150)    # Apply Canny edge detection
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=0.1, param1=80, param2=19, minRadius=5, maxRadius=11)  # Perform Hough Transform to detect circles

        # Draw circles on the original frame
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int") 
            for (x, y, r) in circles:
                # Convert circle coordinates from search region to frame coordinates
                x += center[0]-50
                y += center[1]-50
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

    # Display the frame with detected circles
    cv2.imshow('Problem 2', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()

print('\nProblem 2 - Done!')

'''------------------------------------Problem 3------------------------------------------------------------'''
img = cv2.imread('train_track.jpg')  # Read the image file
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Convert the image to grayscale
blur = cv2.GaussianBlur(gray, (191, 191), 2.5)  # Apply Gaussian blur to reduce noise
thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY)[1]    # Apply a threshold to obtain a binary image
lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=80, minLineLength=850, maxLineGap=1)    # Perform Hough Transform to detect lines
line_img = np.zeros_like(img)   # Create a blank image to draw the lines on

# Find the two lines corresponding to the train tracks
left_line, right_line = None, None
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x2 - x1) > 5 and abs((y2 - y1) / (x2 - x1)) > 0.5: # filter out horizontal and too vertical lines
        if (slope:= x1 - x2) < 0 and (left_line is None or x1 < left_line[0]):  # Find the left line
            left_line = (x1, y1, x2, y2)
        elif (right_line is None or x1 > right_line[0]): # Find the right line
            right_line = (x1, y1, x2, y2)

left_x1, left_y1, left_x2, left_y2 = left_line  
cv2.line(line_img, (left_x1, left_y1), (left_x2, left_y2), (0, 255, 0), 10)    # Draw the left line
right_x1, right_y1, right_x2, right_y2 = right_line
cv2.line(line_img, (right_x1, right_y1), (right_x2, right_y2), (0, 255, 0), 10)    # Draw the right line

height, width = img.shape[:2]
src_pts = np.array([(left_x2, left_y2),(right_x1, right_y1), (left_x1, left_y1),(right_x2, right_y2)], np.float32)  # Define the source points
dst_pts = np.array([[0,0], [height, 0], [0, width], [height, width]], np.float32)       # Define the destination points
M = cv2.getPerspectiveTransform(src_pts, dst_pts)   # The transformation matrix M can be used to warp the image
warped_img = cv2.warpPerspective(img, M, (height, width))   # Warp the image using the transformation matrix M

gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)  # Convert the warped image to grayscale
thresh_warped = cv2.threshold(gray_warped, 250, 255, cv2.THRESH_BINARY)[1]    # Apply a threshold to obtain a binary image
lines_warped = cv2.HoughLinesP(thresh_warped, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)  # Perform Hough Transform to detect lines
line_img_warped = np.zeros_like(warped_img)  # Create a blank image to draw the lines on

distances = [abs(y2 - y1) for line in lines_warped for x1, y1, x2, y2 in line]  # Compute the distances between the lines
for x1, y1, x2, y2 in lines_warped[:, 0]:
    cv2.line(line_img_warped, (x1, y1), (x2, y2), (0, 255, 0), 10)  # Draw the lines
avg_distance = sum(distances) / len(distances)  # Compute the average distance between the lines

# Display the images
fig, axes = plt.subplots(2, 3, figsize=(10, 6)) # Create a figure with a 2x3 grid of Axes
plt.gcf().suptitle('Problem 3 - Average distance between train tracks: {} pixels'.format(avg_distance))  # Set the window title
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(thresh, cmap='gray')
axes[0, 1].set_title('Binary Image')
axes[0, 2].imshow(line_img, cmap='gray')
axes[0, 2].set_title('Lines')
axes[1, 0].imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Warped Image')
axes[1, 1].imshow(thresh_warped, cmap='gray')
axes[1, 1].set_title('Binary Warped Image')
axes[1, 2].imshow(line_img_warped, cmap='gray')
axes[1, 2].set_title('Lines in Warped Image') 
plt.show()

print('Problem 3 - Average distance between train tracks:', avg_distance, 'pixels')

'''------------------------------------Problem 4------------------------------------'''
image = cv2.imread('hotairbaloon.jpg')  # Read the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
blur = cv2.GaussianBlur(gray, (5, 5), 0)    # Apply a Gaussian blur to the grayscale image
_, thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)    # Apply thresholding to create a binary image    
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the binary image

# Draw bounding boxes around each contour and count the number of balloons
count = 0
for contour in contours:
    area = cv2.contourArea(contour) 
    if area > 5000: # Only consider contours with area greater than 5000        
        color = tuple(map(int, np.random.randint(0, 255, size=3)))  # Generate a random color
        x, y, w, h = cv2.boundingRect(contour)  # Get the bounding box coordinates
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Draw the bounding box
        cv2.putText(image, f'Balloon {count}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Label the bounding box
        count += 1

# Display the final image with the count of balloons
cv2.putText(image, f'Total Balloons: {count-1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('Problem 4', cv2.resize(image, (1800, int(image.shape[0] * 1800 / image.shape[1])))) # Resize the image to fit the screen
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Problem 4 - Total Number of balloons:', count,'\n')