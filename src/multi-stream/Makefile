include ../../common/make.config

all: global.o nw_gpu.o nw_cpu.o nw-multi-stream nw-single-stream

global.o: global.c
	$(GCC) $(GCC_FLAGS) -I$(INCLUDE) $^ -c -o $@

nw_gpu.o: nw_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE) $^ -c -o $@

nw_cpu.o: nw_cpu.c
	$(GCC) $(GCC_FLAGS) $^ -c -o $@

nw-multi-stream:	global.o nw_cpu.o nw_gpu.o nw-multi-stream.cu
	$(NVCC) $(NVCC_FLAGS) $^ -L$(CUDA_LIB_DIR) -lcuda -lcudart -I$(INCLUDE) -o $@

nw-single-stream:	global.o nw_cpu.o nw_gpu.o nw-single-stream.cu
	$(NVCC) $(NVCC_FLAGS) $^ -L$(CUDA_LIB_DIR) -lcuda -lcudart -I$(INCLUDE) -o $@

clean:
	rm -f *.o nw-multi-stream
