# L12D2PCANet

README

This is a python implementation of L1-2D2PCANet: A Deep Learning Network for Face Recognition. 

l12dpcanet.py: 
the implementation of the main algorithms. 
To use it just import l12dpcanet.py. 
Call function W=get_l12dpcanet_filters(X), where X is the image array you want to transform, and W is the filter.
Then call function X_final = get_l12dpcanet_features(X,W), the X_final is the deducted dimension image array.

experiment_l12dpcanet.py:
the implementation of the experiment.
To use it just change the DIRECTORY_TRAIN and DIRECTORY_TEST parameters. 
DIRECTORY_TRAIN is the path of the training dataset.
DIRECTORY_TEST is the path of the testing dataset.
The output should be the accuarcy of the face recognization using l12dpcanet.

7404_yale_data_divide.ipynb:
used for divide dataset. 
Change tar_img_folder parameter to change the path you want to store the generated datasets.
It will select i(i=2-7) images from each subject as training set and the rest as test set.

7404_yale_data_noise.ipynb:
used to generate random block-wise noise to the datasets.
Change size parameter to change the size of the noise.

7404_yale_prepare_data.ipynb:
used to pre process the image. Move the face to the center and adjust the pixel size.
Change size parameter to change the size of the image.

