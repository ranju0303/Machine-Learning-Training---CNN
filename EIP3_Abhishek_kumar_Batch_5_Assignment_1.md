![cnn-logo.jpg](https://github.com/ranju0303/Machine-Learning-Training---CNN/blob/master/cnn-logo.jpg?raw=true)
    

# Convolution
The process of extraction features from input data using kernels/filters.Filters move discreetly on top of data/channels.It takes bunch of kernels which applied to given image and creates different activation features in that input image.

- Input - W * H * D
- Kernels/filter - 3 * 3 * D
- No of filters - N
- O/P - W-2 * H-2 * N

It extracts the activation features from given I/P image i.e one filter could be extracting edges, one features could be extracting color and so on.


# Filters/Kernels

Kernel is simply a 2-dimensional matrix of numbers.Kernels are the convolution matrices or masks that is used to detect some features of the image we are processing.Techniques such as blurring, edge detection, and sharpening all rely on kernels.
An image is just a 2-dimensional matrix of numbers, or pixels. Each pixel is represented by a number.For an 8 bit RGB image each pixel has a red, green, and blue component with a value ranging from 0 to 255. A kernel works by operating on these pixel values using straightforward mathematics to construct a new image which is compressed form of that image.

 Filters are used to identify particular object or particular feature from different regions of an image.The filters are in the form of matrices depicting particular attribute. It is applied on top of input image matrix and separates attribute of an image.

# Epoch

An epoch is a complete pass through a given datset. It is number of times the entire dataset gets traversed by the model. An epoch include many iterations.Epoch are required to help neural network get better by giving more and more of same thing repeatetively. As the number of epoch increase so does the model output gets better as long as overfitting is taken cared of.


# 3x3 convolution
3X3 convolution reduces the dimension of the image by 2 dimensions. It reduces the size and dimensions capturing maximum information.consider following examples : 
Consider a grayscale (size 28 X 28 X 1) image showing a single digit number. Our aim is to identify the number. Single digit can be from 0 to 9. Hence, we have 10 possible results.
Every pixel's intensity in the image is represented by a value.

**Note**: In 28 X 28 X1, 1 is the number of channels. It is 1 for grayscale and 3 for RGB images.

![image_3x3.JPG](https://github.com/ranju0303/Machine-Learning-Training---CNN/blob/master/image_3x3.JPG?raw=true)

By rule of thumb, we consider a random 3X3 matrix to slide on this image and extract the features. This results to a layer of 26 X 26. When convoluted again it results in 24 X 24, then 22 X 22 and so on till it reaches 10 X 10 where a well learnt neural network gives the correct output.

- 3 X 3 matrix used here is termed as _'filter'_.
- In the process of convolution when a filter is applied on the input, the result is nothing but sum of dot products of the 2 matrices.


# 1x1 convolution

1X1 convolution is a 1x1 filter that we apply on image which results in same dimensions of the image without loosing much information.
1x1 will make one activation map and each neuron will focus with unique part in the image/input.So this will reduce dimensional and you can later apply the network to any image size .This is one way to compress these feature maps into one.

Example:If you have an input volume of 10x10x12 and you convolve it with a set of D filters each one with size 1x1x12 you reduce the number of features from 12 to D. The output volume is, therefore, 10x10xD

# Feature Maps

The feature map is the output of one filter applied to the previous layer. A given filter is drawn across the entire previous layer, moved one pixel at a time.The output of this convolution operation  is stored in the resulting feature map.
The number of filters (kernel) you will use on the input will result in same amount of feature maps.Feature maps is a mapping of where a certain kind of feature is found in the image.

# Feature Engineering (older computer vision concept)
Feature engineering, the process creating new input features for machine learning, is one of the most effective ways to improve predictive models.In general, you can think of data cleaning as a process of subtraction and feature engineering as a process of addition.This is often one of the most valuable tasks a data scientist can do to improve model performance

# Activation Functions
  
  Activation step apply the transformation to the output by using activation functions where all the negative values are replaced by 0 and only positive values are retained.One of the activation function is Relu.  Further transformation can be done on activation function by applying pooling.Maxpooling takes maximum position value from small region of output to produce single output. It reduces the dimensionality of the feature map. It reduces the parameter model needs to learn.



# How to create an account on GitHub and upload a sample project
How to create GitHub account

  1. Go to www.github.com  
  2. Sign up by providing your Username, email and password field and click on Sign up button (Step 1).
  3. Choose "Unlimited public repositories for free" Plan (Step 2).
  4. You will receive a verification  email from github to the inbox of email id provided.
  5. click on the link provided in the email for confirmation.
  6. Sign in to the github.com
  7. Then you can update your profile details and set up email alerts at different level in "Notification" Tab.
  8. You can upload your project and files in to github. First create a repository by clicking the upper-right corner of any page, click (+) , and then click New          repository.
  9. Enter your repository name ,choose public or private and click create repository button.
  10. Copy the repository link 

Then go to your command prompt, clone the project or github for desktop:
Type following command in CMD console:
    git clone <link>
 Create your project file under this directory.For example: 
    git add sample.txt
    git commit -m "first commit"
    git push 

To Verify:  
        Enter your git username and password.
        Now, you have created your project github. 
        
        
 # Receptive field
  In Convolution small sets of input matrix are connected to the one hidden layer nodes.This region of small sets of input matrix ,which is multiplied with filters and summed up to  get a hidden layer node  ,is referred to as local receptive field.It is used to create feature map from input layer to the hidden layer networks.
  Consider a image having 60 * 60 pixel with 1 channel. If we apply 3*3 filter on each subset of input matrix,what we get is 58*58 matrix with 1 channel.Now the output is one of the identified feature of the actual image.
  
  |input|Filter    |Output|
  |:---------:|:------:|:-------:|
  |60* 60 * 1|3 * 3 * 1|58*58|
  



