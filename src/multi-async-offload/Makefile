include ../../common/make.config

all: global.o nw_phi.o nw_cpu.o nw-async-offload nw-sync-offload

global.o: global.c
	$(ICC) $(ICC_FLAGS) $^ -c -o $@

nw_phi.o: nw_phi.c
	$(ICC) $(ICC_FLAGS) $^ -c -o $@

nw_cpu.o: nw_cpu.c
	$(GCC) $(ICC_FLAGS) $^ -c -o $@

nw-async-offload: global.o nw_cpu.o nw_phi.o nw-async-offload.c
	$(ICC) $(ICC_FLAGS) $^ -o $@

nw-sync-offload: global.o nw_cpu.o nw_phi.o nw-sync-offload.c
	$(ICC) $(ICC_FLAGS) $^ -o $@

clean:
	rm -f *.o nw-multi-offload nw-single-offload
