# ISMI_TIGER
For the ISMI Project we took on the lymphocytes and plasma cells detection task of the TIGER Grand Challenge.

## Detection
This folder contains the necessary files to create a dataset and train on this dataset with a yolov7 model. 

1. **./Detection/ISMI_dataset.ipynb** is the python notebook that was used to create the dataset from the [Zenodo dataset](https://zenodo.org/record/6014422#.ZFo7_s7P2Um). Once the whole notebook is ran the '**cell_dataset**' will be created in this directory and will need to be moved to **./Detection/yolov7** to train the model.

2. **./Detection/tissue-cells** is part of the [Zenodo dataset](https://zenodo.org/record/6014422#.ZFo7_s7P2Um) that is labeled with bounding boxes. This directory is used in **./Detection/ISMI_dataset.ipynb** to create the dataset

3. **./Detection/yolov7** is the yolov7 github that is used to train the pre-trained yolov7 model. One addition to the standard yolov7 github is the **./Detection/yolov7/ISMI_training.ipynb** python notebook that is used to train the yolov7 model. 

## Docker
This folder contains the files necessary to create the docker image that was submitted to the TIGER Grand Challenge leaderboard. This is heavily based on [pathology-tiger-algorithm-example](https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example/tree/main/tigeralgorithmexample)  

1. **./Docker/libs** contains the yolov7 github as well as the trained model weights.

2. **./Docker/test_input** A test .tif image should be placed in this directory and the associated mask should be placed in **./Docker/test_input/images** to test out the algorithm within the docker environment by running the **./Docker/test.sh** script.

3. **./Docker/tigeralgorithmexample** contains the 5 python files that are used to run the docker algorithm. Specifically the **./Docker/processing.py** file contains the relevant code to load the trained yolov7 model and perform detection (no segmentation or TIL score computation.


# Testing the algorithm

To test the algorithm and go through our steps do the following:

## Training the model

1. Run all cells in the **./Detection/ISMI_dataset.ipynb** notebook and move the created '**cell_dataset**' folder inside the **./Detection/yolov7** directory.
2. Train the algorithm by running the relevant cells in the **./Detection/yolov7/ISMI_training.ipynb** notebook.
3. Move the obtained best model weights **best.pt** from **./Detection/yolov7/runs/training/tiger_detection/weights** to **./Docker/libs/weight**.

## Testing the model

0. Be in a linux environment and have docker installed.
1. Add a .tif image and mask in the **./Docker/test_input** folder.
2. Run the **./Docker/test.sh** script to test out the model. 


 
