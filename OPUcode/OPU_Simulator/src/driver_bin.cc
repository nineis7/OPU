#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include "nlohmann/json.hpp"

#include "accelerator.h"
#include "logging.h"
#include "debug_config.h"

void open_debug_file(Device *dev) {
    #ifdef DEBUG_DMA_FM
        dev->OpenOfile(DEBUG_DMA_FM_FILENAME);
    #endif
    #ifdef DEBUG_DMA_KER
        dev->OpenOfile(DEBUG_DMA_KER_FILENAME);
    #endif
    #ifdef DEBUG_OUT_ADDER_A
        dev->OpenOfile(DEBUG_OUT_ADDER_A_FILENAME);
    #endif
    #ifdef DEBUG_OUT_ADDER_B
        dev->OpenOfile(DEBUG_OUT_ADDER_B_FILENAME);
    #endif
    #ifdef DEBUG_PSUM
        dev->OpenOfile(DEBUG_PSUM_FILENAME);
    #endif
    #ifdef DEBUG_PSUM_CUT
        dev->OpenOfile(DEBUG_PSUM_CUT_FILENAME);
    #endif
}

int main(int argc, char*argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;

    Device* dev = new Device();
    // initialize dram
    std::string dram_bin_filename = argv[3];
    dev->dram.base = dev->dram.Alloc(2 << 28);
    dev->dram.FromFile(0, dram_bin_filename);
  #ifdef SIMULATE_INS_RAM
    dev->LoadIns(std::atoi(argv[1]));  // load first 1024 32-bit inst to on-chip inst ram 
  #else
    // initialize instruction ram
    std::string filename = argv[1];
    std::cout << "read from " << filename << "\n";
    dev->ins_ram_.InitFromFile(filename);
  #endif
    open_debug_file(dev);
    // run
    int layer_cnt = std::atoi(argv[2]);
    int sync_cnt = -1;
    if (argc > 4) {
      sync_cnt = std::atoi(argv[4]);  // sync count for last layer to run
    }
    for (int i = 0; i < layer_cnt; i++) {
      int layer_index = i + 1;
      LOG_INFO << "\nLayer " << layer_index << "\n";
      dev->FetchInsn();
      if (i == layer_cnt - 1 && sync_cnt != -1) {
        dev->Run(sync_cnt);
      } else {
        dev->Run();
      }
      if (layer_index == 1) {  // .T
        dev->dram.SaveNumpy(234032, /*row*/512, /*col*/768, /*ofmap fraclen*/13, "b_ofmap_" + std::to_string(layer_index));
      }
      if (layer_index == 2) {
        dev->dram.SaveNumpy(246320, /*row*/512, /*col*/768, /*ofmap fraclen*/11, "b_ofmap_" + std::to_string(layer_index));
      }
      if (layer_index == 3) {
        dev->dram.SaveNumpy(258608, /*row*/512, /*col*/768, /*ofmap fraclen*/12, "b_ofmap_" + std::to_string(layer_index));
      }
      if (layer_index == 4) {
        dev->dram.SaveNumpy(270896, /*row*/512, /*col*/512*12, /*ofmap fraclen*//*14*/15, "b_ofmap_" + std::to_string(layer_index));
      }
      if (layer_index == 5) {
        dev->dram.SaveNumpy(369200, /*row*/512, /*col*/768, /*ofmap fraclen*/13, "b_ofmap_" + std::to_string(layer_index)); 
      }
      if (layer_index == 6) {
        dev->dram.SaveNumpy(381488, /*row*/512, /*col*/768, /*ofmap fraclen*/9, "b_ofmap_" + std::to_string(layer_index));
      }
      if (layer_index == 7) {
        dev->dram.SaveNumpy(393776, /*row*/512, /*col*/3072, /*ofmap fraclen*/11, "b_ofmap_" + std::to_string(layer_index));
      }
      if (layer_index == 8) {
        dev->dram.SaveNumpy(442928, /*row*/512, /*col*/768, /*ofmap fraclen*/11, "b_ofmap_" + std::to_string(layer_index));
      }
    }
    dev->os.close();
    dev->CloseAllOfile();
    return 0;
}