# Fake-News-Classifier
A Binary fake news classifier built as a part of IBM Hackathon, Pravega 2019.

# Instructions

**Note** : glove.6B.100d.txt should be downloaded from [this link](https://drive.google.com/open?id=1Z_1-zH21xT13ciUub1Ttd5v04RH3IXq9) and needs to be pasted in the same directory as the `setup.sh`.

The following files are included in the directory provided:

* `training/`: This folder contains the Fake News Dataset compiled by Perez-Rosas et. al. at the University of Michigan [citation below]. You are required to use only this dataset to train your classifier. There is a README inside this folder to which you can refer for more information.

* `classifier.sh`: You are allowed to write your code in the language of your choice; use this script as a wrapper such that it can be executed from the command line as follows:

    ```./classifier <path to article file>```

    The article file will follow the same format as the articles in the training set (i.e., first line is the title and the rest is the content). This script is required to provide an exit status of 0 if the article is real, and 1 if it is fake. Presently, the exit status is random. As of now, it executes a dummy script (`dummy.py`) that outputs prime numbers upto the input argument on the command line, and exits with a random exit status (0 or 1)
    
    ```
    $ ./classifier 20 \
    2 3 5 7 11 13 17 19
    ```

* `evaluate.sh` : This script will be used to evaluate the time and space complexity of your code.
```
    Eg: $ ./evaluate ./classifier 10000
    Mem  : 7556
    Time : 8.24
```

**Note** : this script suppresses output from your classifier, and reports "Command exited with non-zero status 1" if your classifier reports the article as real, and only outputs the peak memory and total time elapsed between the launch and exit of your classifier.


* `setup.sh`: This file is expected to perform all the environmental setup required to run your code on the provided virtual machine, including installation of all required libraries, and downloading of any special packages (eg: `nltk.download("wordnet"))`. It will be run as:
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

-----


