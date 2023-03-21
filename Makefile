NVCC = /bin/nvcc
CC = g++

GPUOBJS = main.o gpuNetwork.o 
OBJS = cpuNetwork.o arrayUtils.o

network:$(OBJS) $(GPUOBJS)
	$(NVCC) -arch=sm_52 -rdc=true -lcudadevrt -o network $(OBJS) $(GPUOBJS)

main.o: main.cu
	$(NVCC) -c main.cu

gpuNetwork.o: gpuNetwork.cu
	$(NVCC) -arch=sm_52 -rdc=true -lcudadevrt -c gpuNetwork.cu

cpuNetwork.o: cpuNetwork.c 
	$(CC) -c cpuNetwork.c

arrayUtils.o: arrayUtils.c
	$(CC) -c arrayUtils.c


clean:
	rm -f *.o
	rm -f network
