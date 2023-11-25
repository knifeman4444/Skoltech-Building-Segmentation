# Satellite Building Segmentation

*Solution of ThreeNearestNeighbours team*

## General pipeline

### Preprocessing

We split the initial images into chunks of size 256x256 and then apply various augmentations, including 90-degree rotations, horizontal and vertical flips and changing color

For this purpose we created a custom dataset class `SegmentationDataset` which is located in `Utils/segmentation_dataset.py`

`Utils/augmentations.py` contains extended set of augmentations used to reduce overfitting 


### Training

#### Classic Models

We utilize the vast set of classic segmentation models (Unet, Linknet, PAN, DeepLabV3) with different backbones (ResNet, EfficientNet, etc.). 
File `Utils/segmentation_model.py` contains easy-to use training loop implementation with automatic logging.

For rapid experimentation we used `segmentation_models_pytorch` library, which provides easy-to-use API for training and inference.
To track experiments we used Weights&Biases service:

![Weights&Biases](screenshots/wandb.png "Weights&Biases")

The best score was achieved with `Unet++` model with `efficientnet-b5` backbone - `~0.76` for building class and `~1.0` for background class

#### Ensemble

TODO

### GAN approach

TODO


### Reproducing the results

File `pedict.py` contains script with various arguments to run all available models on any data. 
To use it, first execute `get_ds_and_models.sh` script to download all necessary data and models.
Then execute `python predict.py --path_to_pics data/test/images/ --path_to_predictions  data/test/predictions --path_to_models models/ --device cuda` to run all models on test data.

After that, predictions folder will contain subfolders with predictions for all models.

You can also compute mectrics by providing `--path_to_masks` argument with path to ground truth masks


## Results

TODO

## What didn't work

TODO