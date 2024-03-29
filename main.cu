#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "mnistFileUtils.h"
#include "cpuNetwork.h"
#include "arrayUtils.h"
#include "timing.h"

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
    int epochs = 10000;
    static const int amountOfData = 8;



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
                    hiddenTwoLayerBias, outputLayerBias, orInput, inputCorrect, amountOfData, INPUTSIZE, HIDDENSIZE, OUTSIZE, epochs, lr);           
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

        cudaEventRecord(launch_begin,0);
        for(int i = 0; i < epochs; i++) {
            shuffle(orInput, inputCorrect, amountOfData, INPUTSIZE);
            cudaMemcpy(d_input_layer, orInput, d_inputLayer_size * amountOfData, cudaMemcpyHostToDevice);
            cudaMemcpy(d_correct, inputCorrect, (sizeof(float) * OUTSIZE * amountOfData), cudaMemcpyHostToDevice);
            gpuTrainNetwork<<<1, 1>>>(d_input_layer, d_hLayerOne, d_outLayer, d_fWeights, d_outWeights, d_fBias, d_outBias, d_input, d_correct, amountOfData,INPUTSIZE, HIDDENSIZE, OUTSIZE, epochs, lr);    
            cudaDeviceSynchronize();
        }

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

//    printf("Timing simple GPU implementation… \n");
//    // record a CUDA event immediately before and after the kernel launch
//    cudaEventRecord(launch_begin,0);
//    while( 1 )
//    {
    //     // block.x = block.x / 2;
    //    reduce3<<<grid, block, tile_width * sizeof(float)>>>(d_in, d_out, num_in);
    //    check_cuda_errors(__FILE__, __LINE__);
    //    cudaDeviceSynchronize();

    //    // if the number of local sum returned by kernel is greater than the threshold,
    //    // we do reduction on GPU for these returned local sums for another pass,
    //    // until, num_out < threshold
    //    if(num_out >= THRESH)
    //    {
            
    //        num_in = num_out;
    //        num_out = ceil((float)num_out / tile_width);
    //        grid.x = num_out; //change the grid dimension in x direction
    //        //Swap d_in and d_out, so that in the next iteration d_out is used as input and d_in is the output.
    //        temp = d_in;
    //        d_in = d_out;
    //        d_out = temp;
    //     //    tile_width = tile_width / 2;
    //     //    num_in = num_in / 2;
    //    }
    //    else
    //    {
    //        //copy the ouput of last lauch back to host,
    //        cudaMemcpy(h_out, d_out, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
    //        break;
    //    }
    // }//end of while

    // cudaEventRecord(launch_end,0);
    // cudaEventSynchronize(launch_end);

    // // measure the time spent in the kernel
    // float time = 0;
    // cudaEventElapsedTime(&time, launch_begin, launch_end);

    // printf(" done! GPU time cost in second: %f\n", time / 1000);
    // printf("The output from device is:");
    // //if(shouldPrint)
    // printArray(h_out, num_out);

    // // deallocate device memory
    // cudaFree(d_in);
    // cudaFree(d_out);
    // free(h_out);
    // cudaEventDestroy(launch_begin);
    // cudaEventDestroy(launch_end);