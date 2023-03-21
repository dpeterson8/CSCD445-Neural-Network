#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "cpuNetwork.h"
#include "arrayUtils.h"
#include "timing.h"

#include "gpuNetwork.h"

int INPUTSIZE = 2;
int HIDDENSIZE = 2;
int OUTSIZE = 1;

void usage(){
    printf("./project arg1 arg2\n");
    printf("The first argument will determine whether the code si run using cpu or gpu side.\n");
    printf("For cpu side './project 1' will cause the cpu side to run, anthing else will run gpu\n");\
    printf("arg2 will be the number of epochs the network will train with\n");
}

int main( int argc, char *argv[] ) {


    if(!(argc == 2 || argc == 3)) {
        usage();
        exit(1);
    }

    int epochs = 15000;
    if(argv[2] != NULL) {
        epochs = atoi(argv[2]);
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

    float * gpuTotalCorrect =(float *) malloc(sizeof(float));
    int totalCorrect = 0;
    float lr = 0.4;
    static const int amountOfData = 24;



    // iniate hidden layer weights and biases
    iniateWeigts(hiddenLayerWeights, (INPUTSIZE * HIDDENSIZE));
    iniateWeigts(hiddenLayerBias, (HIDDENSIZE));

    // iniate out layer weights and biases
    iniateWeigts(outLayerWights, (HIDDENSIZE * OUTSIZE));
    iniateWeigts(outputLayerBias, (OUTSIZE));
 
    if(*argv[1] == '1') {
        float average_cpu_time = 0;
        clock_t now, then;
        int num_cpu_test = 3;
        unsigned int sum = 0;
        
        then = clock();
        trainNetwork(inputLayer, hiddenOneLayer, hiddenTwoLayer, outLayer, hiddenLayerWeights, hiddenTwoLayerWeights, outLayerWights, hiddenLayerBias,
                    hiddenTwoLayerBias, outputLayerBias, orInput, inputCorrect, amountOfData, INPUTSIZE, HIDDENSIZE, OUTSIZE, epochs, lr, totalCorrect);           
        now = clock();

        float time = 0;
        time = timeCost(then, now);

        average_cpu_time += time;
        average_cpu_time /= num_cpu_test;
        printf(" done. CPU time cost in second: %f\n", average_cpu_time);

    } else {

        cudaEvent_t launch_begin, launch_end;
        cudaEventCreate(&launch_begin);
        cudaEventCreate(&launch_end);

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
        float * d_totalCorrect;
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
        cudaMalloc((void**)&d_totalCorrect, (sizeof(float)));
        cudaMemcpy(d_input_layer, orInput, d_inputLayer_size * amountOfData, cudaMemcpyHostToDevice);
        cudaMemcpy(d_correct, inputCorrect, (sizeof(float) * OUTSIZE * amountOfData), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fWeights, hiddenLayerWeights, hiddenLayerSize * INPUTSIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fBias, hiddenLayerBias, hiddenLayerSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_outWeights, outLayerWights, (sizeof(float) * HIDDENSIZE * OUTSIZE), cudaMemcpyHostToDevice);
        cudaMemcpy(d_outBias, outputLayerBias, outLayerSize, cudaMemcpyHostToDevice);

        cudaEventRecord(launch_begin,0);
        for(int i = 0; i < epochs; i++) {
            shuffle(orInput, inputCorrect, amountOfData, INPUTSIZE);
            cudaMemcpy(d_input_layer, orInput, d_inputLayer_size * amountOfData, cudaMemcpyHostToDevice);
            cudaMemcpy(d_correct, inputCorrect, (sizeof(float) * OUTSIZE * amountOfData), cudaMemcpyHostToDevice);
            gpuTrainNetwork<<<1, 1>>>(d_input_layer, d_hLayerOne, d_outLayer, d_fWeights, d_outWeights, d_fBias, d_outBias, d_input, d_correct, amountOfData,INPUTSIZE, HIDDENSIZE, OUTSIZE, epochs, lr, d_totalCorrect);    
            cudaDeviceSynchronize();
        }

        cudaMemcpy(gpuTotalCorrect, d_totalCorrect, (sizeof(float)), cudaMemcpyDeviceToHost);

        printf("Total correct: %d , Out of: %d\n", (int)gpuTotalCorrect[0], (int)(amountOfData*epochs));

        cudaEventRecord(launch_end,0);
        cudaEventSynchronize(launch_end);

        // measure the time spent in the kernel
        float time = 0;
        cudaEventElapsedTime(&time, launch_begin, launch_end);

        printf(" done! GPU time cost in second: %f\n", time / 1000);

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
