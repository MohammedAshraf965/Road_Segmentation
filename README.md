# Road_Segmentation

A Semantic Segmentation project for road scenes and autonomous driving tasks. The dataset used was KittiDataset, but it can be modified and adapted to the desired data

You can download the dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015). Make sure to create an account since cvlibs require an account to verify the use of their data

# Preparing the Training and Validation Split

- create_data_folder.py: Creates and initializes the folders for the train and validation splits. The dataset has 200 images and masks (including semantic, instance, and rgb). The ratio used was 4:1. The script needs to be executed first so as to create the data folders for the train script
- config.py: Initializes the names and colors for the objects to be detected. Although the dataset has a labels.py which contains all the names and values needed, you can create your own labels and color palettes
- model.py: Initializes the model to be used. The function assumes the use of DeepLabV3 which contains an auxilary classifier that assits in training, it can be discarded if another model achitecture is used. Using 'DEFAULT' for the weights returns the best weights for the model saved by pytorch
- datasets.py: Preparation of the dataset and dataloaders for the training. Keep in mind that this script is to be modified according to the task needed. I have adapted it to the kittidataset, but it can be used for other datasets, though it requires modifications
- utils.py: Helper functions for saving the best model according to the loss and IOU. Other helper functions can be added to assist in other tasks
- engine.py: Contains the training and validation functions
- train.py: The training script that is responsible for:
    - Initializing arguments like the desired image size, batch size, learning rate, epochs, and scheduler (if used)
    - Creating an output directory to save the results (plots and model)
    - Initializing the model, optimizer, and loss function
    - Preparing the training and validation datasets and dataloaders
    - Starting the training loop

To start the training, run the following command:
```bash
python train.py --batch 8 --imgsz 384 --lr 0.05 --epochs 100
```

# Result Analysis
