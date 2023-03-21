#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "mnistFileUtils.h"
#include "cpuNetwork.h"
#include "arrayUtils.h"

#include "gpuNetwork.h"

int INPUTSIZE = 2;
int HIDDENSIZE = 2;
int OUTSIZE = 1;

void usage(){
    printf("./project 1\n");
    printf("The only argument will determine whether the code si run using cpu or gpu side.\n");
    printf("For cpu side './project 1' will cause the cpu side to run, anthing else will run gpu\n");
}

int main( int argc, char *argv[] ) {
    
    // float * inputArr = (float *) malloc(sizeof(float) * 784 * 60000);
    // float * correctInput = (float *) malloc(sizeof(float) * 60000);
    // float * correctData = (float *) (malloc(sizeof(float) * 60000 * 10));

    // getMnistTrain(inputArr, correctInput, correctData, 1);

    // __time_t t;

    if(argc != 2) {
        usage();
        exit(1);
    }
    
    srand((unsigned) time(NULL));
    
    float * inputLayer = (float *) malloc(sizeof(float) * INPUTSIZE);
    float * hiddenOneLayer = (float *) malloc(sizeof(float) * HIDDENSIZE);
    float * hiddenTwoLayer = (float *) malloc(sizeof(float) * HIDDENSIZE);
    float * outLayer = (float *) malloc(sizeof(float) * OUTSIZE);

    float * layers = (float *) malloc(sizeof(float) * INPUTSIZE + (HIDDENSIZE * 2) + OUTSIZE);
    float * weights = (float *) malloc(sizeof(float) * (INPUTSIZE * HIDDENSIZE * 2) + (OUTSIZE * HIDDENSIZE));
    float * biases = (float *) malloc(sizeof(float) * (HIDDENSIZE * 2 + OUTSIZE));

    float * hiddenLayerWeights = (float *) malloc(sizeof(float) * OUTSIZE * INPUTSIZE) ;
    float * hiddenTwoLayerWeights = (float *) malloc(sizeof(float) * HIDDENSIZE * 2);
    float * outLayerWights = (float *) malloc(sizeof(float) * OUTSIZE * HIDDENSIZE);

    float * hiddenLayerBias = (float *) malloc(sizeof(float) * HIDDENSIZE);
    float * hiddenTwoLayerBias = (float *) malloc(sizeof(float) * HIDDENSIZE);
    float * outputLayerBias = (float *) malloc(sizeof(float) * OUTSIZE);

    float orInput[48] = { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                       0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                       0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    float inputCorrect[24] = {0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,
                              0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,
                              0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0}; 


    float lr = 0.4;
    int epochs = 100000;
    static const int amountOfData = 8;



    // iniate hidden layer weights and biases
    iniateWeigts(hiddenLayerWeights, (INPUTSIZE * HIDDENSIZE));
    iniateWeigts(hiddenLayerBias, (HIDDENSIZE));

    // iniate out layer weights and biases
    iniateWeigts(outLayerWights, (HIDDENSIZE * OUTSIZE));
    iniateWeigts(outputLayerBias, (OUTSIZE));

    if(*argv[1] == '1') {
        
        trainNetwork(inputLayer, hiddenOneLayer, hiddenTwoLayer, outLayer, hiddenLayerWeights, hiddenTwoLayerWeights, outLayerWights, hiddenLayerBias,
                    hiddenTwoLayerBias, outputLayerBias, orInput, inputCorrect, amountOfData, INPUTSIZE, HIDDENSIZE, OUTSIZE, epochs, lr);            
    } else {

        float * d_input;
        float * d_correct;
        size_t d_input_size = sizeof(float) * amountOfData;

        float * d_input_layer;
        size_t d_inputLayer_size = sizeof(float) * INPUTSIZE;

        float * d_hLayerOne;
        float * d_hLayerTwo;
        float * d_outLayer;

        float * d_fWeights;
        float * d_outWeights;

        float * d_fBias;
        float * d_outBias;

        float * d_deltaOut;

        float * d_deltaOne; 

        int hiddenLayerSize = sizeof(float) * HIDDENSIZE;
        int outLayerSize = sizeof(float) * OUTSIZE;

        int hiddenWeightSize = sizeof(float) * HIDDENSIZE * INPUTSIZE;

        cudaMalloc((void**)&d_input, d_input_size);
        cudaMalloc((void**)&d_correct, (sizeof(float) * OUTSIZE * amountOfData));
        cudaMalloc((void**)&d_input_layer, d_inputLayer_size * amountOfData);
        cudaMalloc((void**)&d_fWeights, hiddenWeightSize);
        cudaMalloc((void**)&d_fBias, (sizeof(float) * HIDDENSIZE));
        cudaMalloc((void**)&d_hLayerOne, hiddenLayerSize * amountOfData);
        cudaMalloc((void**)&d_hLayerTwo, hiddenLayerSize);
        cudaMalloc((void**)&d_outWeights, (sizeof(float) * HIDDENSIZE * OUTSIZE));
        cudaMalloc((void**)&d_outBias, (sizeof(float) * OUTSIZE));
        cudaMalloc((void**)&d_outLayer, outLayerSize * amountOfData);    
        cudaMalloc((void**)&d_deltaOut, (sizeof(float) * OUTSIZE) * amountOfData);
        cudaMalloc((void**)&d_deltaOne, (sizeof(float) * HIDDENSIZE) * amountOfData);

        cudaMemcpy(d_input_layer, orInput, d_inputLayer_size * amountOfData, cudaMemcpyHostToDevice);
        cudaMemcpy(d_correct, inputCorrect, (sizeof(float) * OUTSIZE * amountOfData), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fWeights, hiddenLayerWeights, hiddenLayerSize * INPUTSIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fBias, hiddenLayerBias, hiddenLayerSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_outWeights, outLayerWights, (sizeof(float) * HIDDENSIZE * OUTSIZE), cudaMemcpyHostToDevice);
        cudaMemcpy(d_outBias, outputLayerBias, outLayerSize, cudaMemcpyHostToDevice);

        for(int i = 0; i < epochs; i++) {
            shuffle(orInput, inputCorrect, amountOfData, INPUTSIZE);
            cudaMemcpy(d_input_layer, orInput, d_inputLayer_size * amountOfData, cudaMemcpyHostToDevice);
            cudaMemcpy(d_correct, inputCorrect, (sizeof(float) * OUTSIZE * amountOfData), cudaMemcpyHostToDevice);
            gpuTrainNetwork<<<1, 1>>>(d_input_layer, d_hLayerOne, d_outLayer, d_fWeights, d_outWeights, d_fBias, d_outBias, d_input, d_correct, amountOfData,INPUTSIZE, HIDDENSIZE, OUTSIZE, epochs, lr);    
            cudaDeviceSynchronize();
        }

        cudaFree(d_input);
        cudaFree(d_correct);
        cudaFree(d_input_layer);
        cudaFree(d_fWeights);
        cudaFree(d_fBias);
        cudaFree(d_hLayerOne);
        cudaFree(d_hLayerTwo);
        cudaFree(d_outWeights);
        cudaFree(d_outBias);
        cudaFree(d_outLayer);
        cudaFree(d_deltaOut);
        cudaFree(d_deltaOne);

    }

    free(inputLayer);
    free(hiddenOneLayer);
    free(hiddenTwoLayer);
    free(outLayer);
    free(layers);
    free(weights);
    free(biases);
    free(hiddenLayerWeights);
    free(hiddenTwoLayerWeights);
    free(hiddenLayerBias);
    free(hiddenTwoLayerBias);
    free(outputLayerBias);

}