from setuptools import setup
from setuptools import find_packages

setup(name='ml_project2',
      version='1.0',
      description='Project 2: Road segmentation PRML',
      author='Kaicheng Yu, Jan Band',
      install_requires=['numpy', 'scipy', 'pillow', 'keras'],
      extras_require={
          'cnn_train': ['theano'],
          'traditional': ['sklearn']
      },
      packages=find_packages())