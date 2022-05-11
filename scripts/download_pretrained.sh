#!/bin/bash

echo "Downloading pretrained models..."

if gdown 118jgmOq1hFRYGIHq-i1gX6lHU2vuDsMm; then
  unzip shape_as_points.zip
  mv shape_as_points out
  rm shape_as_points.zip
  echo "Done!"
else
  echo "Please install gdown with 'conda install -c conda-forge gdown'"
fi