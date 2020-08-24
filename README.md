# SMM4H
This repository contains our CNN that we propose for Task 2 of SMM4H 2020. Task 2 of SMM4H 2020 is the binary classfication of tweets that contain ADEs. The dataset is highly imbalanced and we propose 3 methods to address that: oversampling, desampling, and Keras class weights. 

## Install
To install clone this repository using Git:
``` 
git clone https://github.com/NLPatVCU/SMM4H.git 
```
Then, create a virtual enviorment. You should use Python 3.6. 
``` 
python3 -m venv env
source env/bin/activate 
pip install -r requirements.txt
```

## Overview 
Data is preprocessed and extraneous information is removed. Then, it can be passed through the Unbalanced class where it can be desampled or oversampled. After that, the data goes through the Model class where it is prepared for the CNN. Finally, the CNN is run. At this stage, there is an option to do CV or train-test, use Test data, or use Keras class weights. 
## Running Experiments 
To run an experiment: 
```
python experiments.py
```
In the experiments.py file, there is an example and comments for other possible options. 
## Docs
For more detailed documentation, check out: https://smm4h.readthedocs.io/en/latest 
  
