NVCC = /bin/nvcc
CC = g++

GPUOBJS = gpuNetwork.o
OBJS = main.o cpuNetwork.o arrayUtils.o mnistFileUtils.o


network:$(OBJS) $(GPUOBJS)
	$(NVCC) -arch=sm_52 -o network $(OBJS) $(GPUOBJS)

main.o: main.c
	$(CC) -c main.c

gpuNetwork.o: gpuNetwork.cu
	$(NVCC) -arch=sm_52 -c gpuNetwork.cu

cpuNetwork.o: cpuNetwork.c 
	$(CC) -c cpuNetwork.c

arrayUtils.o: arrayUtils.c
	$(CC) -c arrayUtils.c

mnistFileUtils.o: mnistFileUtils.c
	$(CC) mnistFileUtils.c

clean:
	rm -f *.o
	rm -f network
