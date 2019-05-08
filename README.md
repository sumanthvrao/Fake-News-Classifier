# Fake-News-Classifier
A Binary fake news classifier built as a part of IBM Hackathon, Pravega 2019.

Code Repository: <a target="_blank" href="https://github.com/sumanthvrao/Fake-News-Classifier" rel="noreferrer noopener">https://github.com/sumanthvrao/Fake-News-Classifier</a>

Report: <a target="_blank" href="Report_PPT.pdf" rel="noreferrer noopener">Presentation.pdf</a>

Built a news classification system to determine the authenticity of a news article by considering the scores from a binary classification model (LSTM Network with Ensemble Learning) and a fact-checking model. Used IBM Watson's Natural Language Understanding API to design a system capable of scoring every news article based on its authenticity.

## Tech Stack
* IBM Watson’s NLP APIs
* Python - Keras and Pandas libraries
* Bing Web Search API

## Setup and Execution Details

**Note** : glove.6B.100d.txt should be downloaded from [this link](https://drive.google.com/open?id=1Z_1-zH21xT13ciUub1Ttd5v04RH3IXq9) and needs to be pasted in the same directory as the `setup.sh`.

The following files are included in the directory provided:

* `training/`: This folder contains the Fake News Dataset compiled by Perez-Rosas et. al. at the University of Michigan [citation below]. This dataset to train your classifier. There is a README inside this folder to which you can refer for more information.

* `classifier.sh`: This script is a wrapper such that it can be executed from the command line as follows:

    ```./classifier <path to article file>```

    The article file will follow the same format as the articles in the training set (i.e., first line is the title and the rest is the content). This script is provides an exit status of 0 if the article is real, and 1 if it is fake.

    ```
    $ ./classifier ./training/fakeNewsDataset/legit/entmt01.legit.txt
    ```

* `evaluate.sh` : This script will be used to evaluate the time and space complexity of your code.
```
    Eg: $ ./evaluate ./classifier ./training/fakeNewsDataset/legit/entmt01.legit.txt
    Mem  : 7556
    Time : 8.24
```

* `setup.sh`: This file performs all the environmental setup required to run the code, including installation of all required libraries, and downloading of any special packages (eg: `nltk.download("wordnet"))`. It will be run as:
```
    $ sudo -H ./setup.sh
```
----

### Dataset citation:

@article{Perez-Rosas18Automatic, <br/>
author = {Ver\’{o}nica P\'{e}rez-Rosas, Bennett  <br/>Kleinberg, Alexandra Lefevre, Rada Mihalcea}, <br/>
title = {Automatic Detection of Fake News}, <br/>
journal = {International Conference on  <br/>Computational Linguistics (COLING)}, <br/>
year = {2018} <br/>
}

## Team
* Sumanth V Rao
* Sumedh Pb
* Suraj Aralihalli
* Tejas Prashanth