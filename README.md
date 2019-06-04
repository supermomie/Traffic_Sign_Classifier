# **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./demo/pltdistrib.png "Visualization"
[image2]: ./demo/demoprepros1.png "preprocess1"
[image3]: ./demo/demoprepros2.png "preprocess2"
[image4]: ./demo/demoprepros3.png "preprocess3"
[image5]: ./demo/plthistory.png "history diag"
[image6]: ./demo/respred2.png "respred"
[image7]: ./demo/S.png "example pred"
[image8]: ./demo/vizuconv1.png "visuconv1"
[image9]: ./demo/vizuconv2.png "visuconv2"


---


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...
As we can see some classes have a larger number of training images belonging to them.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to convert the images to grayscale because for differentiating such images from any other sort of color image is that less information needs to be provided for each pixel. After that I decide to equilize fot improves the contrast in the image.

Here is an example of a traffic sign image before and after grayscaling.

1 plot the normal pic ![alt text][image2]
2 change the color map for less info ![alt text][image3]
3 equilise it ![alt text][image4]

This is all I need.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


```python
def trafficNet(x):
    
    # 3x3 convolution with ReLU activation
    conv1 = conv2d(x, 3, 1, 12)
    conv1 = tf.nn.relu(conv1)
    
    # 3x3 convolution with ReLU activation
    conv1 = conv2d(conv1, 3, 12, 24)
    conv1 = tf.nn.relu(conv1)
    pool1 = maxPool(conv1, 2)
    
    # 5x5 convolution with ReLU activation
    conv2 = conv2d(pool1, 5, 24, 36)
    conv2 = tf.nn.relu(conv2)
    
    # 5x5 convolution with ReLU activation
    conv2 = conv2d(conv2, 5, 36, 48)
    conv2 = tf.nn.relu(conv2)
    pool2 = maxPool(conv2, 2)
    
    # Flatten
    fc0   = flatten(conv2)
    
    # Dropout
    fc0 = tf.nn.dropout(fc0, 1)
    
    # First Fully
    fc1 = fully(fc0, 1728, 512)
    fc1 = tf.nn.relu(fc1)

    # Second Fully
    fc2 = fully(fc1, 512, 256)
    fc2 = tf.nn.relu(fc2)

    # Output layer
    logits = fully(fc2, 256, num_classes)
    
    return logits, conv1, conv2
```


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| preprocess picture   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 1x1x12 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 12x12x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 24x24x36 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 36x36x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 3x3x48  				|
| Flatten				|												|
| Dropout				|	1											|
| Fully connected		| Input 1728 Output 512							|
| RELU					|												|
| Fully connected		| Input 512 Output 256							|
| RELU					|												|




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a little batch size (20) for more precision. The number of epochs is at most 50 but during the training it is less 50, it cut automaticly if the train accuracy does not evolve (no evolution beyond 5 epochs)

![alt text][image5]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of +99%
* validation set accuracy of +/-97%
* test set accuracy of +/-94%


My first Archictecture was like this

I decide to start by the 1x1 convolution, much easiest for this learning I think

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| preprocess picture   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 1x1x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x12 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 12x12x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x36   				|
| Flatten				|												|
| Dropout				|	1											|
| Fully connected		| Input 4356 Output 512							|
| RELU					|												|
| Fully connected		| Input 512 Output 256							|
| RELU					|												|

I decide to start with this architecture for check the performance, after that I decide to add more convolution layer et see what happend. The valid accuracy was +/-91%, it's a good result but not enought. So I add 2 new conv2D and change the order, by starting with a 2 convolutions 3x3 then 2 convolutions 5x5.
I customized the different parameters like outputs's convolution and the first fully to finally get a valid accuracy to +/-97%.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I choose some really difficult picture for the test, with some writing on them, slope and a distance

![alt text][image6]

The model predict 8 on 10 pictures, it'a good start.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the some results of predictions with top 5 softmax probabilities:

![alt text][image7]


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

bis_2

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


![alt text][image8]
![alt text][image9]

