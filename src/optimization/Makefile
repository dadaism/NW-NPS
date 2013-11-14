include ../../common/make.config

all: nw_gpu.o nw_cpu.o nw-nps

nw_gpu.o: nw_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE) $^ -c -o $@

nw_cpu.o: nw_cpu.c
	$(GCC) $(GCC_FLAGS) $^ -c -o $@

nw-nps:	nw_cpu.o nw_gpu.o nw-nps.cu
	$(NVCC) $(GCC_FLAGS) $^ -L$(CUDA_LIB_DIR) -lcuda -lcudart -I$(INCLUDE) -o $@

debug: nw_cpu.o nw_gpu.o nw-nps.cu
	$(NVCC) $(GCC_FLAGS) -D DEBUG $^ -L$(CUDA_LIB_DIR) -lcuda -lcudart -I$(INCLUDE) -o $@

traceback: nw_cpu.o nw_gpu.o nw-nps.cu
	$(NVCC) $(GCC_FLAGS) -D TRACEBACK $^ -L$(CUDA_LIB_DIR) -lcuda -lcudart -I$(INCLUDE) -o $@

clean:
	rm nw_cpu.o nw_gpu.o debug traceback nw-nps
