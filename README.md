# Project 2 Road Extraction from satellite image
PRML project 2 Road Extraction


## Running requirement and installation
In order to run the ```run.py```, please make sure you have successfully
installed all the required libraries and set the corresponding paths correctly.

### 1. Saved weights and meta data
Since our best model is trained in TensorFlow, you will need the obtain complete saved weights to get the evaluation results. You could get the trained weights and related testing files through the link

Unzip the file, you will get xxx

### 2. Installing required libraries

To only reproduce the best submission csv file and corresponding prediction images, you only need to install the following libraries and python 2.7 interpreter. It is strongly recommended to create a virtual environment to install these libraries so that you could easily delete them afterwards.

* Keras
```pip install keras```
* tensorflow
```pip install tensorflow```
* Other python packages
```
'matplotlib',
'numpy',
'scipy',
'pillow',
'cPickle'
```

In order to run all of the other models described in the documentation, following libraries are needed:

* Theano (if not already installed with Keras)
```pip install theano```
* scikit-learn
```pip install -U scikit-learn```
* scikit-image
```pip install scikit-image```



### 3. Environment vairable setting

Finally, go to the project root folder (the directory contains this README.md file) and
execute
```bash
python setup.py
```

Then, run the
```bash
python project2/run.py
```

You should be able to see the results in no time, depending on the computational power :)
Normally, for a Macbook Pro, it should take approximately
On a 32-core CPU machine, it takes around 120 seconds.

## baseline

To run the gradient boosting + random walker algorithm, go in the folder project2/baseline and run the jupiter notebook called Gradient boosting and random walker.ipynb.

## Project directory structure
Please download the run-time 
```
path/to/project/
    project2/   # source codes
        ...
        run.py      # Run routine to reproduce the best result
        ...
    
    tensorboard/    # Folder stored the tensorflow stored model
        xxx/            # xxx is the model name, to be set in run.py
        
    output/     # Output folder
    
    data/       # Data folder
        massachuttes/       # generated dataset
            label/              # label image
            sat/                # sattellite image
        training/           # training dataset
            groundtruth/        # label image
            images/             # sattellite image
        test_set_images/    # testing dataset
            test_sat/           # sattellite image
            test_label/         # empty folder
    
    model/      # Save-path to vgg19 model
    
```

## Obtaining the dataset
As explained in the report large amount of data were needed for training FCN. Therefore, in order to reproduce the training process (~28 hours on GPU GeForce GTX TITAN Z) you need to download the Massachusetts Road and Building Detection Dataset and preprocess it as described in following steps.

### Obtaining the dataset
The aerial images: https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html
The ground truth masks: https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html

### Preprocessing
Adjust variables *inputPath* and *outputPath* in INPUT PARAMETERS part of the script **utils/images2patches.py** so that the paths would correspond to your directory structure. All other parameters should be left as is (in order to compy with the description in the report). Then run the script, the resulting images will be stored in *outputPath*

### Directory structure
Move the generated images into *data/massachusetts/sat/* and the generated masks into *data/massachusetts/label* directories.

