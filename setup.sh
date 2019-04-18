#!/bin/bash

apt-get install python3-pip
pip3 install wheel
pip3 install setuptools
pip3 install nltk pandas sklearn
python3 -c 'import nltk; nltk.download("wordnet"); nltk.download("stopwords"); nltk.download("all");'
pip3 install --upgrade watson-developer-cloud
pip3 install language_check
pip3 install textstat
pip3 install statistics
pip3 install six
pip3 install numpy scipy
pip3 install pillow
pip3 install h5py
pip3 install --upgrade tensorflow
pip3 install Keras

