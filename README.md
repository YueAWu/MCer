# MCer

## Paper and Data
Our WWW 2021 paper: [Multi-level Connection Enhanced Representation Learning for Script Event Prediction](https://dl.acm.org/doi/10.1145/3442381.3449894).

We uploaded data to [google drive](https://drive.google.com/file/d/1rZgDE8djN717xYE8em0OoFRir9hv8uxq/view?usp=sharing). You need to download and unzip it to root directory.

## Execute the code
You can run the following codes in console for model training:

`python train_model.py MCer 0.0 0.05 0.1 1e-8 0.01 2000 10 100` 

`python train_model.py MCerLSTM 0.2 0.05 5e-5 1e-7 0.1 2000 10 200` 

MCer/MCerLSTM is model name and the following numbers are the optimal hyperparameters (dropout, margin, learning rate, weight decay, momemtum, batch size, epochs, patients).

We saved model parameters that are better than the results in our paper. You can run the following code in console for model testing:

`python test_model.py`

Note: you have to parallelize the code on two GPUs.

## Requirements
* Python 3.6
* PyTorch 1.3.0
* numpy
* sklearn
* GPU (Tesla V100 or P100 is recommended)
