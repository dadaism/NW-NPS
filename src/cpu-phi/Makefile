include ../../common/make.config

all: global.o nw_phi.o nw_cpu.o nw-cpu-phi

global.o: global.c
	$(ICC) $(ICC_FLAGS) $^ -c -o $@

nw_phi.o: nw_phi.c
	$(ICC) $(ICC_FLAGS) $^ -c -o $@

nw_cpu.o: nw_cpu.c
	$(ICC) $(ICC_FLAGS) $^ -c -o $@

nw-cpu-phi:	global.o nw_cpu.o nw_phi.o nw-cpu-phi.c
	$(ICC) $(ICC_FLAGS) $^ -o $@

clean:
	rm *.o nw-cpu-phi
