# CS763 Traffic Light Detection

## Data
The [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset)
will be used for this project. This dataset contains over 10K images of traffic lights 
in an urban environment along with associated bounding boxes for detection. Please 
follow the instructions below to download and set up the data. Use 7-Zip to unzip only 
the `.001` file of each dataset (see [here](https://hiro.bsd.uchicago.edu/node/3168) for 
more information).
1. Download only the RGB files of the train (4 files) and test (7 files) datasets.
2. Extract the training data from `dataset_train_rgb.zip.001`. It is split across 7 
subdirectories starting with a datetime. Also, extract `train.yaml` for bounding boxes.
3. Extract the testing data from `dataset_test_rgb.zip.001`. It is all stored in a 
single directory. Also, extract `test.yaml` for bounding boxes.
4. Move the datasets into the `data/images/train` and `data/images/test` directories.
The YAML files are already provided but can be replaced if the dataset has been updated.
5. Confirm there are 5093 training images and 8334 testing images.
6. Run `data_utils.py` to convert the provided bounding box labels into the Pascal VOC
and YOLO data format. Additionally, a validation subset will be created.
