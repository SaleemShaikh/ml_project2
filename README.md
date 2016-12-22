# Project 2 Road Segmentation from satellite image
Team KPJ, by Kaicheng Yu, Pierre Runavot, Jan Bednarik
PRML Course project 2, EPFL

We present the approaches we used to tackle the task of road segmentation and
the results we achieved. We utilize the well known computer vision methods,
among which the gradient boosting and random walker algorithm proved to bring
fruitful results, as well as currently very popular deep learning in the means
of both the standard CNN architectures and very recent state-of-the-art models
known as fully convolutional networks (FCN). With the random walker algorithm,
AlexNet CNN architecture and FCN8 architecture we achieved 80%, 81% and 92%
accuracy respectively on the originally provided test dataset.

## Running requirement and installation
In order to run the ```run.py```, please make sure you have successfully
installed all the required libraries and set the corresponding paths correctly.
There should be at least 4 GB free space on your machine.
### 1. Download required files
Since our best model is trained in TensorFlow, you will need the obtain complete
saved weights to get the evaluation results. You could get the trained weights
and related testing files through the link
https://drive.google.com/open?id=0B-YlxKup3Jzfd1JENXp6b0tGeVE
Unzip the file, you will get the following folders
```
path/to/download/
  kpj_reproduce.zip
  kpj_reproduce/      # Unzipped directory, move to working dir
      data/
      model/
      tensorboard/
```
Move or copy the folders to current working directory (a.k.a. the directory
  contains this README.md)
The complete directory of project should look like section Project structure

### 2. Installing required libraries

To only reproduce the best submission csv file and corresponding prediction
images, you only need to install the following libraries and python 2.7
interpreter. It is strongly recommended to create a virtual environment to
install these libraries so that you could easily delete them afterwards.

* Keras
```
pip install keras
```
* tensorflow
```
pip install tensorflow
```
* Other python packages, you could also use pip install.
```
'matplotlib',
'numpy',
'scipy',
'pillow',
'cPickle'
```

In order to run all of the other models described in the documentation, following libraries are needed:

* Theano (if not already installed with Keras)
```
pip install theano
```
* scikit-learn
```
pip install -U scikit-learn
```

### 3. Environment variable setting

Finally, go to the project root folder (the directory contains this README.md
  file) and execute

```bash
python setup.py
```
If you cannot run the
code due to 'ImportError', try to execute

```bash
export PYTHONROOT=/path/to/project/dir:$PYTHONROOT
python project2/run.py
```

## HOWTO run and interpret results


Run the
```bash
cd /path/to/kpj/  # path you extracted from kpj_submission.zip
python project2/run.py
```

You should be able to see the results in no time, depending on the
computational power :) Normally, for a Macbook Pro, it should take approximately
5 minutes. On a 32-core CPU machine, it takes around 200 seconds.

Generate result is inside ```output``` folder, with following structure

## HOWTO train FCN model

On a CUDA enable machine, the rough running time for fit one image is 0.5
seconds, with batch size 4, 5,000 iterations take 10,000 seconds which is more
than 2 hours.

In order to train, there are two basic pipeline.

### 1. Training on Original Dataset only
If no pre-training, you could simply download the training data,
https://inclass.kaggle.com/c/epfml-segmentation/download/training.zip
then extract them into ```path/to/project/data/```
They should be in exactly the same name as project directory structure.

Then, open the ```train_fcn.py``` and follows the instructions to set accordingly

A sample setting is
```python
# Set CUDA visible device masks
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# Set Keras backend and image dimension ordering
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_IMAGE_DIM_ORDERING'] = 'tf'

# Model name, plot directory, iterations and fine-tune name
MODEL_NAME = 'fcn8s_test_train'
PLOT_DIR = 'plot_test'
MAX_ITERATION = int(5000 + 1)
MODE = 'finetune'
FINETUNE_NAME = 'test_finetune'
```

Finally, run ```python project2/train_fcn.py``` to initialize the training on
original data.

### 2. Pre-train on generated dataset

If you want to pre-train the model, please firstly go to download massachuttes
dataset first, then follows section "Obtaining the generated dataset". It will
results in 12,000 (400x400) sat images and label images under
directory  ```data/massachuttes/```.

Then, alter the ```train_fcn.py``` ```MODE``` to ```'train'```.

To complete pre-train on the dataset, you need approximately 24 hours on a Titan
Z GPU.

After the pre-train is done, one folder with name ```MODEL_NAME``` will be created
under ```tensorboard``` folder. It could be further fine-tune.

### 3. Fine-tune on model

After obtaining pre-train model, we could specify a ```FINETUNE_NAME``` to obtain
better results.

Follows the step 1 like setting, please set the ```MODEL_NAME``` the same as
step 2 and then give a ```FINETUNE_NAME```. Set the ```MODE``` to ```finetune```

Then run the ```train_fcn.py``` again.

To get better result, please turn the flag to ```True```
```
tf.flags.DEFINE_bool('augmentation', 'True', 'Data runtime augmentation mode : True/ False')
```

Complete step 2 - 3 takes approximately 28 hours and will occupy around 10 GB on
your disk.


## Project directory structure
The complete project directory
```
path/to/project/
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

    output/     # Output folder

    project2/   # source codes
        ...
        run.py      # Run routine to reproduce the best result
        ...

    tensorboard/    # Folder stored the tensorflow stored model
        xxx/            # xxx is the model name, to be set in run.py

    kpj_report.pdf
    README.md
    LICENSE
```

## Obtaining the generated dataset
As explained in the report large amount of data were needed for training
FCN. Therefore, in order to reproduce the training process (~28 hours on
 GPU GeForce GTX TITAN Z) you need to download the Massachusetts Road
 and Building Detection Dataset and preprocess it as described
 in following steps.

### Obtaining the dataset
The aerial images: https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html
The ground truth masks: https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html

### Preprocessing
Adjust variables *inputPath* and *outputPath* in INPUT PARAMETERS part
of the script **utils/images2patches.py** so that the paths would
correspond to your directory structure. All other parameters should be
left as is (in order to compy with the description in the report). Then
run the script, the resulting images will be stored in *outputPath*

### Directory structure
Move the generated images into *data/massachusetts/sat/* and the
generated masks into *data/massachusetts/label* directories.
