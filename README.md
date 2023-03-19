# MID-TERM
## ENPM673 - Perception for Autonomous Robots

## Dependencies
1. python 3.11 (any version above 3 should work)
2. Python running IDE (I used VS Code)

## Libraries
1. OpenCV
2. NumPy
3. Matplotlib

## Contents
1. ball.mov
2. train_track.jpg
3. hotairbaloon.jpg
4. vishnum_midterm.py
5. vishnum_midterm.pdf
6. README.md
7. problem2_output.png
8. problem3_output.png
9. problem4_output.png

## Installation Instructions
1. Download the zip file and extract it
2. Install python and the required dependencies: pip install opencv-python numpy matplotlib

## Overview
This code implements solutions for three different computer vision problems using OpenCV and Python. The problems are:

1. Detecting a ball in a video and drawing circles around it
2. Detecting train tracks in an image and transforming it into a top-view with the tracks parallel to each other and finding the average distance between them.
3. Counting the number of balloons in an image by detecting contours and drawing bounding boxes around them.

## Problem 2
### Description
This problem involves detecting a ball in a video and drawing circles around it. The color thresholding technique is used to filter the red channel and locate the center of the ball. A search region is defined around the center, and edge detection is performed on this region. Hough Transform is used to detect circles in the search region, and the circles are drawn on the original frame.

### Usage
1. Place the video file in the same directory as the code file
2. Set the filename of the video in the cv2.VideoCapture() function call in line 6 of the code
3. Run the code: vishnum_midterm.py. It will display the video with circles drawn around the ball. Press q to exit the video.

### Output
![Ball Hough Circle](https://github.com/vishnumandala/Hough-Transformation-Perspective-Transformation-and-Contour-Detection/blob/main/problem2_output.png)

## Problem 3
### Description
This problem involves detecting train tracks in an image and drawing lines along them. Gaussian blur is applied to the grayscale image to reduce noise. A threshold is applied to obtain a binary image. Hough Transform is used to detect lines in the binary image, and the left and right lines corresponding to the train tracks are identified. These lines are drawn on a blank image and then warped using perspective transform to obtain a top-down view of the tracks. Hough lines are drawn on the warped image and distance between them is calculated to find the average distance between the tracks.

### Usage
1. Place the image file in the same directory as the code file
2. Set the filename of the image in the cv2.imread function call in line 53 of the code
3. Run the code: vishnum_midterm.py. It will display the original image with the lines drawn along the tracks. The warped image will also be displayed. The average distance between tracks is computed and displayed in the terminal. Press any key to exit.

### Example Output
Problem 3 - Average distance between train tracks: 2280.5555555555557 pixels

### Output
![Perspective Transformation of Train Track](https://github.com/vishnumandala/Hough-Transformation-Perspective-Transformation-and-Contour-Detection/blob/main/problem3_output.png)

## Problem 4
### Description
This problem aims to count the number of balloons in an image and draw bounding boxes around each balloon. Gaussian blur is applied to the grayscale image to reduce noise. A threshold is applied to obtain a binary image. Contours are detected in the binary image and bounding boxes of random colors are drawn around contours with area greater than 5000. They are also labelled and the total count of the balloons is printed.

### Usage
1. Place the image file in the same directory as the code file
2. Set the filename of the image in the cv2.imread function call in line 111 of the code
3. Run the code: vishnum_midterm.py. The program will display the image file with bounding boxes of different colors around each balloon and a count of the total number of balloons will be printed on the terminal.

### Example Output
Problem 4 - Total Number of balloons: 17 

### Output
![Hot Air Balloon Count using Contour Detection](https://github.com/vishnumandala/Hough-Transformation-Perspective-Transformation-and-Contour-Detection/blob/main/problem4_output.png)
