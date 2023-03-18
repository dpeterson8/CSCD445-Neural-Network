NVCC = /bin/nvcc
CC = g++

GPUOBJS = reduce.o 
OBJS = main.o cpuNetwork.o arrayUtils.o mnistFileUtils.o


network:$(OBJS) 
	$(CC) -o network $(OBJS) 

network.o:
	$(NVCC) -arch=sm_52 -c  

main.o: main.c cpuNetwork.c arrayUtils.c mnistFileUtils.c
	$(CC) -c main.c cpuNetwork.c arrayUtils.c mnistFileUtils.c

clean:
	rm -f *.o
	rm -f network
