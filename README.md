## README

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./pipeline.ipynb"  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text](/examples/calibration.PNG)

Using multiple images of chessboard taken through the camera I calibrate the camera. The distortion co-efficient and the matrix required to undistort images are stored to the disk. This helps us undistort images without having to re-calibrate our camera each time

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like these:
![alt text](/examples/distortion-corrected.PNG)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image 
```python
def apply_threshold(img, s_thresh=(20, 255), sx_thresh=(10, 255)):
    l_thresh = (225, 255)
    
    
    ksize = 5
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float)
    b_channel = lab[:,:,2] 
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize, ) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binary = sxbinary
        
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(10, 255))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh=(10, 255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    
    # Get the RGB channels individually
    if len(img.shape) > 2 and img.shape[2] == 4:
        r, g, b, a = cv2.split(img)
    else:
        r, g, b = cv2.split(img)

    
    # Empty channel
    null_channel = np.zeros_like(s_channel)

    # Detect yellow color
    yellow = np.zeros_like(r)
    yellow[( (r>150)&(g>130)&(b>60) & (b<120) )] = 1
    
    # Detect white color
    white = np.zeros_like(r)
    white[( (r>150)&(g>150)&(b>150) )] = 1
       
    detection = np.dstack((yellow, white, null_channel))   
    
    detect = np.zeros_like(s_channel)
    # Yellow or White
    detect[( ((yellow == 1) & (s_binary == 1)) | (( white == 1 ) & (sxbinary == 1)) )] = 1
    color_binary = detect
    
    
    return color_binary

```


Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text](/examples/threshold.PNG)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

LaneDetector python class in my notebook contains the warp method that is used to do perspective transform of the image. 

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. I chose the hardcode the source and destination points in the following manner:

```python
# Image dimension
x = img.shape[1]
y = img.shape[0]

x_mid = x/2

src = np.float32([
        [x_mid - 0.2 * x_mid, 0.7 * y],
        [x_mid + 0.2 * x_mid, 0.7 * y],
        [x_mid + 0.5 * x_mid, 0.9 * y],
        [x_mid - 0.5 * x_mid, 0.9 * y]])


dst = np.float32([
        [offset, offset],
        [x - offset, offset],
        [x - offset, y],
        [offset, y]])

# Computing the perspective transform
M = cv2.getPerspectiveTransform(src, dst)

# To do the reverse
Minv = cv2.getPerspectiveTransform(dst, src)

# Do the actual transformation - using linear interpolation
warped = cv2.warpPerspective(img, M, (x, y), flags=cv2.INTER_LINEAR)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 512, 504      | 200, 200      | 
| 768, 504      | 1080,200      |
| 960, 648      | 1080,720      |
| 320, 648      | 200, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text](/examples/transform.PNG)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Within the python notebook I defined a function `fit_lane_lines` where I have fit my lane lines with a 2nd order polynomial. Using histogram one finds the peak points and using those points as an initial point of the lane lines the rest of the plot is fit


![alt text](/examples/histogram.PNG)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

```python
def measure_curvature(self, ploty, leftx, rightx):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.min(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/900 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

```python
def draw_lanes_on_image(self, image, warped, ploty, left_fit, right_fit, Minv):        
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # If the image has a transparency channel 
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)    

    return result
```

  Here is an example of my result on a test image:

![alt text](/examples/lanes.PNG)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/kR2CiKvoFP4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

The pipeline under performs when there are shadows and changing light intensity of the roads. It also suffers when the roads contain lane markings that were later painted on top. Bumpy roads also causes the lane detection to suffer since the pipeline assumes the road surface to be flat. 
Improving the threshold function and perspective transform could result in better lane detection