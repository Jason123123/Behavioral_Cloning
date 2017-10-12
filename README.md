#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/net.png "Model Visualization"
[image2]: ./examples/drive.jpg "Grayscaling"
[image3]: ./examples/recover1.jpg "Recovery Image"
[image4]: ./examples/recover2.jpg "Recovery Image"
[image5]: ./examples/recover3.jpg "Recovery Image"
[image6]: ./examples/recover3.jpg "Normal Image"
[image7]: ./examples/flip.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network following structure:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   | 
| Normalize | |
| Crop | Crop 50 rows pixels from the top of the image and 20 rows pixels from the bottom of the image|
| Convolution 5x5x24  | 1x1 stride, valid padding |
| ELU					|						|
| Max pooling	   | 2x2 stride   |
| Convolution 5x5x36	| 1x1 stride, valid padding |
| ELU		|        						|
| Max pool		| 2x2 stride        |
| Convolution 5x5x48	| 1x1 stride, valid padding |
| ELU		|        						|
| Max pool		| 2x2 stride        |
| Convolution 3x3x64	| 1x1 stride, valid padding |
| ELU		|        						|
| Max pool		| 2x2 stride        |
| Flatten	|	|
| Dropout| keep_prob = 0.5      |
| Dense 1 |	Output = 100  |
| Dropout| keep_prob = 0.5      |
| Dense 2 |  Output = 64 |
| Dense 3 |  Output = 1 |

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 77 and 79). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the training data provided in the course link.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use multi-layer CNN to predict the proper steering angle of the vehicle.

My first step was to use a convolution neural network model similar to LeNet.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model does not driving well as the car drifts away from the lane center. 

To predict steering angle better, I add more convolutional layer to the model as well as a dropout layer to prevent overfitting. 

After such improvements, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is described in previous session. 

Here is a visualization of the architecture (note: layer size may not be exactly the same)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to keep driving along lane center. These images show what a recovery looks like starting from the right side of the track :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images so that the car won't keep moving toward one side of the road. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had nearly 35000 number of data points. I then preprocessed this data by normalizing them to be zero-centered and cropped the upper part of the image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The model is trained for 5 epoch. As I tested it in the autonomous mode, the car drives nicely without hitting the boundary of the track. I used an adam optimizer so that manually training the learning rate wasn't necessary.
