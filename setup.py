# from setuptools import setup
# from setuptools import find_packages
#
# setup(name='ML_project2',
#       version='1.0',
#       description='Project 2: Road segmentation PRML',
#       author='Kaicheng Yu, Jan Band',
#       install_requires=['numpy', 'scipy', 'pillow', 'keras'],
#       extras_require={
#           'cnn_train': ['theano'],
#           'traditional': ['sklearn']
#       },
#       packages=['project2'])


import os
import sys

# set project path
PROJECT_DIR = os.getcwd()

try:
    sys.path.index(PROJECT_DIR)
except ValueError:
    sys.path.append(PROJECT_DIR)
    print("Add current path to Environment Variable {}".format(PROJECT_DIR))
    print("Environment path ")
    print(sys.path)
    raise ValueError

print("Found path, please execute the run.py inside ./project2 folder by "
      "python project2/run.py")


# List of packages to be installed
package_requires = ['tensorflow', # numpy, six will be installed
                    'keras',
                    'pillow',
                    'matplotlib'
                    ]
print("Please install the following packages with \"pip install\" {}".format(package_requires))
