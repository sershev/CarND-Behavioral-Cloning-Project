**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model_images/model.jpg "Model Visualization"
[image2]: ./IMG/center_2017_02_05_20_58_05_875.jpg "Training image example center cam view."

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network weights
* model.json containing a convolution neural network model
* data_equalizer.py script to get better training data distribution
* config.py config script
* data.py data loading (generator) script
* preprocess_image.py script to preprocess the images
* writeup_report.md this file with report
* environment.yml template for anaconda env

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

#### 3. Submssion code is usable and readable

The model.py file (together with other *.py files) contains the code for training and saving the convolution neural network (Nvidia model with dropout layers). The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

After experimenting with different CNN architectures I decided to go for [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
My model consists of a convolution neural network, which starts with normalization layer, followed by 5 Conv. layers with subsampling parameters (pooling) and each of them is followed by an Activation layer (elu) and an Dropout layer with 0.5 pobability. As next the network have a Flatten layer followed by 2 full conected (Dense) layers where each of them is followed by Activation layer (elu) and an Dropout layer (0.5 prob.). At the end one Dense layer followed by elu activation (this time without Dropout) and output layer (Dense(1)) with linear activation. This all is in **create_model() in model.py**.

#### 2. Attempts to reduce overfitting in the model

The model contains many dropout layers in order to reduce overfitting.
The training date is each time selected randomly from the whole dataset.

The model was trained and validated on different data sets in combination with data augmentation to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually 
```
model.compile('adam', 'mse', ['accuracy'])
```
(['accuracy'] was not really used for the mesurement, instead I looked more to *mse*, and how the car behaves after training.)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving data, provided by Udacity, together with self collected data with the second (beta) simulator.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to 
1. read/research
2. try
3. fail
4. go back to 1.

My first step was to use a convolution neural network model with only one Conv. layer and 3 Dense layers, I thought this model might be trained fast and it was, but it also was unable to drive the car (underfitting problem).

In order to get better result I wrote a function which gives me a better distribution of the training data. I found that the accuray score doesn't really reflect the accuracy of the car.

To combat the overfitting, I added lot of Dropout layers and fed the model slowly with random data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I removed upper 50px of image and converted it to HSV . After it was only failing in two places I started to tweak the model with data only for the desired steering angle 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py create_model()) consisted of a convolution neural network based on [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
(See section 1.3)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![CNN model used for behavior cloning.][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first drived some laps to get better feeling for the simulator. After I used Udacity's data to see how it woukd work. Then I recorded some laps on track one using center lane driving. Here is an example image of center lane driving:

![Example image of cener lane.][image2]

Then I went for beta simulator to get data with smoother steering angles.
Then I recorded some data on track 1 backward and track 2.

To augment the data sat, I also flipped images and angles thinking that this would lead to better data distribution / training result.

After the collection process, I had ~17000 origin (without augmentation) number of data points. I then preprocessed this data by equalizing the destribution, removing top 50px and convert to HSV. I also tried another methods but they seems don't addect the training proccess much.

I finally randomly shuffled the data set. 

I used this training data for training the model. The mse value helped determine if the model was over or under fitting. The ideal number of epochs was about 20, but I also somtimes took only 1 to finetune the model. I used an adam optimizer so that manually training the learning rate wasn't necessary, since adam optimizer have an adoptive learning rate.