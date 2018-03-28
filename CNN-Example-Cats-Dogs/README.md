This directory contains the code to construct and test a simple CNN.  It ouputs a model which takes an image and determines if it's of a cat or a dog. It was put together by following along with these blog posts:
https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/
https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

The images can be downloaded at https://www.kaggle.com/c/dogs-vs-cats/data, and should be saved in a folder 'kaggle_dogs_vs_cats'

To create the model, run: python train_network.py --dataset kaggle_dogs_vs_cats --model cat_dogs.model

Where 'cat_dogs.model' is whatever you want to name the model.

After creating the model, you can test it on any image with: python test_network.py --model cat_dogs.model --image imageName
