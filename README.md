This program contains all the Needleman-Wunsch implementation in our paper:
* **["A Distributed CPU-GPU Framework for Pairwise Alignments on Large-Scale Sequence Datasets" (ASAP '13)](http://www.danielbit.com/dali/papers/CPU_GPU_NW_ASAP13.pdf)**

Directory structure
---------------------------
* bin: generated binary
* common: configuration file for make
* run: script for running
* src: source code
* utility: utility source code

Source file
--------------------------
stat.c:
	a small tool to get distribution of characters in dataset

nw_cpu.c:
	the CPU version

nw_gpu.cu:
	the GPU version

needle_kernel_rodinia.cu:
	kernel function from original Rodinia benchmark

needle_kernel_rodinia_opt.cu:
	optimized kernel fromt original Rodinia benchmark

needle_kernel_tile.cu:
	tiled scan kernel (TDScan)

needle_kernel_diagonal.cu:
	diagonal scan kernel (DScan)

needle_kernel_dynamic.cu:
	row scan kernel (RScan)


FAQ
--------------------------
Q1: How to compile the code?
A1: Just "make" it! But you may need to modify the Makefile to set correct environment variables.

Q2: What are these generated binaries?
A2: nw-nps    - Only calculate scorematrix 
    debug     - Calculate scorematrix with validation on CPU
    traceback - With traceback
	stat      - Utility for get statistic information of dataset

Q3: How to run the code?
A3: Usage: ./nw-nps <max_rows/max_cols> <penalty> 
	<length>  - x and y length
	<pair number>  - number of pairs
	<penalty> - penalty (negative integer)
	<memory> - 0: unpined 1: pinned 2: double buffering
	<kernel> - 0: Rodinia 1: Optimized Rodinia 2: DScan 3: TDScan 4: RScan


