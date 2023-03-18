#ifndef MNISTFILEUTILS_H_
#define MNISTFILEUTILS_H_

void swap(float *a, float *b);
void randomizeArray(float * arr, float * correctArr, int arrSize);
void getMnistTrain(float * inputArr, float * correctInput, float * correctData, int option);

#endif