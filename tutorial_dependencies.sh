#!/bin/bash

echo "Installing OpenCV"
conda install -c conda-forge opencv python=2.7.15
echo "Installing Numpy"
conda install -c anaconda numpy python=2.7.15
echo "Installing Shapely"
conda install -c conda-forge shapely python=2.7.15
echo "Installing Caffe"
conda install -c conda-forge caffe python=2.7.15 numpy=1.15.4
conda install -c anaconda numpy=1.15.4 python=2.7.15

echo "Dependencies Installed!"
