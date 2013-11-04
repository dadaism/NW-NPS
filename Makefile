include common/make.config

ROOT_DIR = $(shell pwd)
SRC = $(ROOT_DIR)/src
BIN = $(ROOT_DIR)/bin

all:
	cd src;		make;	mv nw-nps $(BIN);	rm -f *.o;
	cd utility;	make;	mv stat   $(BIN);	rm -f *.o;

clean:
	cd bin;	rm -f *;
