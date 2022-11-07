all: compile

fastrun:
	./build/sim-json ../Compiler-Backend/linecode.json 1 ../Compiler-Backend/dram_data.bin

run:
	GLOG_v=1 ./build/sim-json ../Compiler-Backend/linecode.json 1 ../Compiler-Backend/dram_data.bin 2>&1 | tee trace.txt
	
compile:
	cd build;make -j4;cd ..
