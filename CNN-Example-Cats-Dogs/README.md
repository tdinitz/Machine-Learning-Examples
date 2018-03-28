This directory contains the code to construct and test a simple CNN.  It ouputs a model which takes an image and determines if it's of a cat or a dog.

The images can be downloaded at https://www.kaggle.com/c/dogs-vs-cats/data, and should be saved in a folder 'kaggle_dogs_vs_cats'

To create the model, type: python train_network.py --dataset  kaggle_dogs_vs_cats --model cat_dogs.model

Where 'cat_dogs.model' is whatever you want to name the model.

After creating the model, you can test it on any image with: python test_network.py --model santa_not_santa.model --image imageName
