
# objectDetection
The project consists of image files taken from **ImageNet** for training and testing and python scripts to get image features with the **ResNet50** model. With these features, a **Feed Forward Neural Network** (FFN) is trained and then tested against 100 cropped versions of each image for candidate window extraction and labeling.

Images will be classified as the following 10 classes:

 - Eagle
 - Dog
 - Cat
 - Tiger
 - Starfish
 - Zebra
 - Bison
 - Antelope
 - Chimpanzee
 - Elephant

## Requirements/Dependencies
Before running the scripts, be sure to have the required packages installed. To install the necessary packages you may run:

    pip install -r requirements.txt
or

    pip3 install -r requirements.txt
## Pre-Processing
To get the **ResNet50** features from the images, normalization is necessary as the **ResNet50** model only accepts (224 x 224) sized images. Images will be padded and resized as (224  x 224) before feature extraction. 

**pre_processing.py** script is used throughout the different stages of the project for normalization.
## Pre-Training
**pre_training.py** script when run, is going to get labels and features for each image for training stage. 

*Note*: saved features and labels for images are currently stored in the **lib** directory

To start the pre-training you may run the script as:

    python pre_training.py
or 

    python3 pre_training.py

When the script runs and finishes, features and labels will be stored in the **lib** directory titled as:

 - **features_train.pt**
 - **features_test.pt**
 - **labels_train.json**
 - **labels_test.json**
 
 ## Training
 **training.py** script when run, is going to train a FFN model with features and labels saved in the pre-training stage. 
 
 *Note*: saved and trained model is currently stored in the **lib** directory as **model.pt** with 95% accuracy and ~0.0005 loss.
 
 To start training you may run the script as:
 
    python training.py
or 

    python3 training.py
    
When run, script is going to print iteration, loss and accuracy information in the terminal for each 100 iterations.
## Testing
To start testing you may run **main.py** python script which is going to call **testing.py**. While running, script is going to print the currently tested filename in the terminal. This process may take a while depending on the machine hardware.

To start testing you may run the script as:

    python main.py
or

    python3 main.py

After script's execution is finished, a new directory titled as **results** will be created where all the testing images with candidate windows and labels are stored in the PNG format titled as **result_FileName.png** (for ex: *result_42.png* for 42.jpeg)

