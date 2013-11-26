include common/make.config

ROOT_DIR = $(shell pwd)
SRC = $(ROOT_DIR)/src
BIN = $(ROOT_DIR)/bin

all:
	cd src/optimization;		make;	mv nw-nps $(BIN);	rm -f *.o;
	cd src/multi-stream;		make;	mv nw-multi-stream $(BIN);	rm -f *.o;
	cd src/cpu-gpu;				make;	mv nw-cpu-gpu $(BIN);	rm -f *.o;
	cd src/multi-async-offload;	make;	mv nw-cpu-gpu $(BIN);	rm -f *.o;
	cd utility;	make;	mv stat   $(BIN);	rm -f *.o;

clean:
	cd bin;	rm -f *;
	cd src/optimization;		make clean;
	cd src/multi-stream;		make clean;
	cd src/cpu-gpu;				make clean;
