#!/bin/bash

#!/bin/bash

./nw-cpu-gpu -k 0 -c 1 -g 1 -l 32 --debug
./nw-cpu-gpu -k 0 -c 10 -g 100 -l 128 --debug
./nw-cpu-gpu -k 0 -c 10 -t 96 -l 512 --debug
./nw-cpu-gpu -k 0 -g 20 -t 96 -b 16 -l 1600 --debug

./nw-cpu-gpu -k 1 -c 1 -g 1 -l 32 --debug
./nw-cpu-gpu -k 1 -c 10 -g 100 -l 128 --debug
./nw-cpu-gpu -k 1 -c 10 -t 96 -l 512 --debug
./nw-cpu-gpu -k 1 -g 20 -t 96 -b 16 -l 1600 --debug

