# VehicleReidentification
The task of re-identifying vehicles across a network of surveillance cameras with non-overlapping fields of view is a challenging and intriguing problem especially in the domain of Transportation Systems. Matching a particular vehicle across multiple camera views in a network is the aim of vehicle re-identification (re-id) project. However, this task is not only challenged by significant intra-class variations and nuanced inter-class differences of vehicles across multiple cameras, but also by the intricate settings of urban surveillance scenario.

## Introduction and Project Overview
This project aims to train the model with images from the 20 different cameras from different settings. During testing, the task to re-identify the match for the query images against gallery. Models are evaluated using the Cumulative Matching Characteristic (CMC) curve and mean Average Precision (mAP) metrics.
The CMC curve is a way of measuring the performance of a vehicle re-identification system by plotting the cumulative probability that a correct match is found in the top-k ranked results. The x-axis represents the rank (i.e., the position in the ranked list of results) and the y-axis represents the cumulative probability. The higher the curve, the better the performance of the model. The mAP metric is another way of evaluating the performance of a vehicle re-identification system, which takes into account both precision and recall. It measures the average precision across all possible thresholds for matching.

## To perform the extraction of vehicles from the overlapping camera images from the gallery set the project enables the below class files

a. src \models – Initialize the models architecture to train the model
b. data_manager – It has the below python class, ImageDataset - To load the custom dataset RandomIdentitySampler - Randomly samples the images for training.
transforms.py - to perform the DataAugmentation like random erasing, color augmentation and random2D translation
BaseDataManager – Has properties likes pids(image id) and camera ids along with the function for loading custom dataset using train loader, test loader to load the train, query and gallery images
ImageDataManager – Extends the base class and load the images to train and test data loader class along with the camera ids
c. Metrics: eval_metrics.py – Evaluate the CMC curve and precision to find the distance metrics between the query and gallery images
d. Loss – Calculate cross entropy and triplet loss
e. Schedulers - lr_schedulers.py – Set the scheduler for single or multi step along with step size and gamma parameter
f. Optimizers - optimizers.py – to enable the optimizer choice (adam, amsgrad, sgd, rmsprop) based on the argument
g. utils:
Averagemeter – to compute the average and value of loss
Iotools – directory verification to save and load the files
RankLogger – records the rank1/5/10/20 matching accuracy
Torchtools – Save the checkpoint, resume the checkpoint, load_pretrained weights to load the pretrained models

## Inputs
The list of input files and images has been provided as input to the project including,
a. image_train – 37,778 images of vehicles
b. image_test – 11,579 images
c. query – 1678 images
d. name_query, name_test, name_train text files – holds the name of the images for query, train and test
e. camera distance, camera Id’s – file holds the information about the source camera
f. list type, list color – holds the color and type of the vehicle
g. gt_index, jk_index –index information
h. test_label.xml/ train_label.xml – contains critical information about the vehicle type, camera id, vehicle id and image name.

## Training the model

Steps:
a. Load the datasets
Trian loader – pull all the images under the train folder also find the number of camera ids and pids (decide the class number)
Test_Loader_Dict – set the gallery images from the test image folder and the query images from the query folder
b. Initialize the model
For pretrained models – should pass the load pretrain weights param along with placing the model weights in code folder, it help to load the specific model.
c. Set the loss to Cross entropy and triplet loss
d. Set the scheduler and optimizer hyperparameters
e. Run the forward step of the model and find the prediction. Based on the prediction, compute loss.
f. Backpropagate the loss and save the gradients

## Testing the model
a. For each image in the query set, find the prediction. Extract the query set features
b. For each image in the test set(gallery) find the prediction with the model. Extract the gallery set features.
c. Find the distance matrix
d. Evaluate the CMC curve and mAP(average precision) to find the matrix
e. Finally show the performance summary for overall results.

## Experiments
In this work, extensive analysis has been performed on the Veri dataset by using various convolutional neural network like Resnet 18, 34, 50, 101, 152, mobilenet and vgg16. Multiple output from the different network has been compared and contrasted to find the reason behind the results and the performance of the networks.
Also, the input has been transformed and augmented to monitor the impacts on the dataset. Finally, different hyperparameters like learning rate, epochs, optimizers, schedulers have been varied to see the optimal performance.