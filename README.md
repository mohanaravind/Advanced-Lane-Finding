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
    ksize = 15
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
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
    detect[( ((yellow == 1) & (s_binary == 1)) | (( white == 1 ) & (sxbinary == 1)) |
           ((gradx == 0) & (grady == 0) & ((mag_binary == 1) & (dir_binary == 1))) )] = 1
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


Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:


![alt text](/examples/histogram.PNG)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text](/examples/lanes.PNG)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://www.youtube.com/watch?v=-8XL7lEfBA4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
