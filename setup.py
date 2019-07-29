#nsml: anibali/pytorch:cuda-9.0
# Setup file for pytorch 4.* 
from distutils.core import setup

setup(name='', version='0.1', install_requires=['numpy',
                                                'pillow',
                                                'torchvision',
                                                'tensorboard',
                                                'matplotlib',
                                                'tensorboard-logger',
                                                'tqdm',
                                                'sklearn',
                                                'tensorflow',
                                                'easydict',
                                                'cython'])
