# CSCD 445 Neural Network
## Objective - 

For are final project in CSCD 445 or GPU Computation we chose to create a neaurel network, the network will use cuda and batching to make it able to train and test data with the same math equtaions used by both CPU and GPU. We will then put togther time comparision as well as some accuracy between the too in search of improvments. By default and currenlt the network runs on only the logical OR statment ex: (1,0) or (0,1) and returns false when (1,1) or (0,0). Another goal is to get the network to learn and recognize the MNIST dataset.

## Requirements to run -
The only required packages for this project is the MNIST dataset in csv format which can be found at:
    https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

## To run and comile

- make -- Will compile project
- ./network -- Will run the compiled code
- make clean -- Will clean the repo of all .o packages as well as the "network" package
