# **Self driving car - Behavioral Cloning**

---

In this project, I trained a self driving car in simulator environment using human driving data.

| Track 1 | Track2 |
| :----: | :----: |
| [![Track 1](https://img.youtube.com/vi/kmcRsvL4OpE/0.jpg)](http://www.youtube.com/watch?v=kmcRsvL4OpE) | [![Track 2](https://img.youtube.com/vi/fnzufSZ4NU0/0.jpg)](http://www.youtube.com/watch?v=fnzufSZ4NU0)  |

[//]: # (Image References)

[image1]: ./figures/first_track_steer_dist.png "Distribution plot of training set for track 1"
[image2]: ./figures/second_track_steer_dist.png "Distribution plot of training set for track 2"
[image3]: ./figures/three_cams.png "Sample of three cameras"
[image4]: ./figures/preprocessing.png "Sample of preprocessing stages"

---

## Included files

My project includes the following files:
* model.py  containing the script to create and train the model
* drive.py  for driving the car in autonomous mode
* model.h5  containing a trained convolution neural network for track 1
* model_track2.h5   containing a trained convolution neural network for track 2
* behavior_clone.ipynb  jupyter notebook to visualize each step
* util.py   for utility functions
* video.mp4 video export of simulation on track 1. Youtube links to video exports for both tracks can be found above.


## Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
To train new file run
```sh
python model.py
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

I used [Nvidia's model](http://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with slight simplifications.
(model.py lines 61-84)

| Layer (type)            |     Output Shape       |       Param    |
|:------|:----|:----|
| conv2d_16 (Conv2D)     |      (None, 40, 160, 24)  |     1824   |
| conv2d_17 (Conv2D)        |   (None, 20, 80, 36)    |    21636     |
| conv2d_18 (Conv2D)       |    (None, 10, 40, 48)   |     43248     |
| conv2d_19 (Conv2D)    |      (None, 10, 40, 64)   |     27712     |
| conv2d_20 (Conv2D)       |    (None, 10, 40, 64)  |     36928     |
| flatten_4 (Flatten)     |     (None, 25600)    |         0         |
| dense_13 (Dense)      |       (None, 100)      |        2560100   |
| dense_14 (Dense)       |      (None, 50)       |         5050      |
| dense_15 (Dense)       |      (None, 10)        |        510       |
| dense_16 (Dense)      |       (None, 1)       |          11        |
| | | |
| Total params: 2,697,019.0 | Trainable params: 2,697,019.0  | Non-trainable params: 0.0 |

The model includes RELU layers to introduce nonlinearity, and the data is cropped and normalized in the model using Keras layers (code line 73).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 97-104).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

#### Appropriate training data

I used provided data set for the first track. I ran and recorded two laps for the second track.

### Model Architecture and Training Strategy

#### Solution Design Approach

My main focus on the solution was to create a model that can learn from imperfect data, without need for excessive data tuning.

I first inspected the training data. I used the provided data for track 1. The data was strongly skewed, as steering measurements were near zero for most of samples. For track 2, I collected data by driving around the track for 2 laps.
Steering measurements were relatively evenly distributed for the track 2 training set.

![Dist_track1][image1]
![Dist_track2][image2]

The image was converted to YUV format, following the methods of Nvidia. Indeed, the model failed to complete track 2 when trained with RGB images. To mitigate imbalanced distribution in track 1 training set, I applied random amounts of shear to the images. The data set was flipped randomly to balance out left and right turns. Top and bottom part of the image was cropped to exclude
unnecessary parts such as sky and car bonnet. Lastly, the data was normalized.

![Preproc][image4]

I also used images from left and right camera as training data. I added small offset value to steering measurement associated with the images, then appended them to training data as center image.

I used Nvidia's self driving car model as it is a proven model for the task. I simplified it slightly to reduce training cost. I trained the model for 4 epochs.  The vehicle is able to autonomously drive around the track without leaving the road.
I used same procedure for track 2, except for that shear was not applied to track 2 data.

#### Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track two, trying best to stay on center lane using mouse, but the data was noisy.
After the collection process, I had 15990 number of data points for track 2. The data were fed to network using custom batch generator.
Data were randomly shuffled and 20% of the data were used as validation set.

I used this training data for training the model. The validation set helped to check for signs of overfitting, but the value it self did not correlate with actual driving quality, understandable since the training data was noisy.

### Result
The model can drive through both tracks successfully.

### Discussion
The model proved to be robust enough to finish the task with relatively noisy data. One interesting point was that the model performed much better with YUV data. Perhaps this was because YUV space divides brightness and color, making the differentiation of road easier.