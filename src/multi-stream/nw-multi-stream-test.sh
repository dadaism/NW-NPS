#!/bin/bash

./nw-multi-stream -k 0 -n 1 -l 40 --debug
./nw-multi-stream -k 0 -n 10 -n 100 -l 128 --debug
./nw-multi-stream -k 0 -n 10 -t 96 -l 512 --debug
./nw-multi-stream -k 0 -n 20 -t 96 -b 16 -l 1600 --debug

./nw-multi-stream -k 1 -n 1 -l 32 --debug
./nw-multi-stream -k 1 -n 10 -n 100 -l 128 --debug
./nw-multi-stream -k 1 -n 10 -t 96 -l 512 --debug
./nw-multi-stream -k 1 -n 20 -t 96 -b 16 -l 1600 --debug

