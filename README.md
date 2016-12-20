# Project 2 Road Extraction from satellite image
PRML project 2 Road Extraction

## Running requirement and installation
In order to run the ```run.py```, please make sure you have successfully
installed all the required library and set the corresponding paths correctly.

### Saved weights and meta data
Since our best model is trained in TensorFlow, you will need the complete
saved weights to get the evaluation results.
You could get the link
Our run-time
* Keras


## Project directory 

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
