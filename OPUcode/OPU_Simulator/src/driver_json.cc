#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include "nlohmann/json.hpp"

#include "accelerator.h"
#include "logging.h"
#include "debug_config.h"

//#define VIT
#define BERT

int run_one_line(std::string line, Device* dev) {
    nlohmann::json j = nlohmann::json::parse(line);
    std::string opcode;
    j.at("opcode").get_to(opcode);
    if (opcode == "load") {
        auto load = dev->load_ins_;
        int load_type;
        std::vector<int> ddr_load_type;
        j.at("ddr_load_type").get_to(load_type);
        if (load_type & 0x1) {
            j.at("ddr_fm_addr_ini").get_to(load->ddr_fm_addr_ini);
            j.at("ddr_load_block_y_size").get_to(load->ddr_load_block_y_size);
            j.at("ddr_load_block_x_size").get_to(load->ddr_load_block_x_size);
            j.at("ddr_fm_read_num").get_to(load->ddr_fm_read_num);
            j.at("fm_in_y_size").get_to(load->fm_in_y_size);
            j.at("fm_in_x_size").get_to(load->fm_in_x_size);
            j.at("fm_bank_id").get_to(dev->fm_ram_id);
            ddr_load_type.push_back(OPU_MEM_ID_FM);
        }
        if (load_type & 0x2) {
            j.at("ddr_ker_addr_ini").get_to(load->ddr_ker_addr_ini);
            j.at("ddr_ker_read_num").get_to(load->ddr_ker_read_num);
            j.at("wgt_bank_id").get_to(dev->wgt_ram_id);
            ddr_load_type.push_back(OPU_MEM_ID_WGT);
        }
        if (load_type & 0x4) {
            j.at("ddr_bias_addr_ini").get_to(load->ddr_bias_addr_ini);
            j.at("ddr_bias_read_num").get_to(load->ddr_bias_read_num);
            j.at("bias_bank_id").get_to(dev->bias_ram_id);
            ddr_load_type.push_back(OPU_MEM_ID_BIAS);
        }
        if (load_type & 0x8) ddr_load_type.push_back(OPU_MEM_ID_RESIDUAL);
        j.at("ddr_load_single").get_to(load->ddr_load_single);
        if (load_type & 0x10) {
            j.at("ddr_ins_addr_ini").get_to(load->ddr_ins_addr_ini);
            ddr_load_type.push_back(OPU_MEM_ID_INS);
        }
        if (load_type & 0x20) {
            j.at("ddr_param_addr_ini").get_to(load->ddr_param_addr_ini);
            j.at("ddr_param_read_num").get_to(load->ddr_param_read_num);
            j.at("param_bank_id").get_to(load->param_ram_id);
            ddr_load_type.push_back(OPU_MEM_ID_PARAM);
        }
        if (load_type & 0x40) {
            j.at("ddr_fm_addr_ini").get_to(load->ddr_fm_addr_ini);
            j.at("ddr_load_block_y_size").get_to(load->ddr_load_block_y_size);
            j.at("ddr_load_block_x_size").get_to(load->ddr_load_block_x_size);
            j.at("ddr_fm_read_num").get_to(load->ddr_fm_read_num);
            j.at("fm_in_y_size").get_to(load->fm_in_y_size);
            j.at("fm_in_x_size").get_to(load->fm_in_x_size);
            j.at("psum_write_addr_s").get_to(load->psum_addr_s);
            ddr_load_type.push_back(OPU_MEM_ID_PSUM);
        }
        load->ddr_load_type = ddr_load_type;        
        dev->RunLoad(load);
    } else if (opcode == "store") {
        auto store = dev->store_ins_;
        j.at("ddr_save_pos").get_to(store->ddr_save_pos);
        j.at("ddr_save_des").get_to(store->ddr_save_des);
        j.at("residual").get_to(store->residual);
        j.at("activation").get_to(store->activation);
        j.at("activation").get_to(dev->reg_.activation);
        j.at("activation_type").get_to(store->activation_type);
        j.at("pooling").get_to(store->pooling);
        j.at("pooling").get_to(dev->reg_.pooling);
        j.at("pooling_type").get_to(store->pooling_type);
        j.at("pooling_x_size").get_to(store->pooling_x_size);
        j.at("pooling_y_size").get_to(store->pooling_y_size);
        j.at("pooling_x_stride").get_to(store->pooling_x_stride);
        j.at("pooling_y_stride").get_to(store->pooling_y_stride);
        j.at("ddr_save_block_x_size").get_to(store->ddr_save_block_x_size);
        j.at("ddr_save_block_y_size").get_to(store->ddr_save_block_y_size);
        //j.at("block_pool_x_size").get_to(store->block_pool_x_size);
        //j.at("block_pool_y_size").get_to(store->block_pool_y_size);
        j.at("fm_output_addr_ini").get_to(store->fm_output_addr_ini);
        j.at("ddr_save_fm_num").get_to(store->ddr_save_fm_num);
        j.at("padding").get_to(store->padding);
        j.at("padding_size").get_to(store->padding_size);
        j.at("fm_out_x_size").get_to(store->fm_out_x_size);
        j.at("fm_out_y_size").get_to(store->fm_out_y_size);
        j.at("channel_out").get_to(store->channel_out);
        j.at("upsample_output").get_to(store->upsample_output);
        store->out_y_max = 0;
        store->out_y_min = 0;
        store->out_x_max = 0;
        store->out_x_min = 0;
        store->out_y_stride = 0;
        store->out_x_stride = 0;
        j.at("nvm_input_fraclen").get_to(store->nvm_input_fraclen); 
        j.at("nvm_output_fraclen").get_to(store->nvm_output_fraclen); 
        if (j.contains("nvm_beta_fraclen")) {
            j.at("nvm_beta_fraclen").get_to(store->nvm_beta_fraclen); 
            j.at("nvm_gamma_fraclen").get_to(store->nvm_gamma_fraclen);
        }
        dev->RunStore(store);
    } else if (opcode == "compute") {
        auto compute = dev->compute_ins_;
        j.at("fm_bank_id").get_to(compute->fm_ram_id);
        j.at("wgt_bank_id").get_to(compute->wgt_ram_id);
        if (j.at("add_bias"))
            j.at("bias_bank_id").get_to(compute->bias_ram_id);
        j.at("type").get_to(compute->type);
        j.at("dw_flag").get_to(compute->dw_flag);
        j.at("ker_x_size").get_to(compute->ker_x_size);
        j.at("ker_y_size").get_to(compute->ker_y_size);
        j.at("dma_block_x_size").get_to(compute->dma_block_x_size);
        j.at("dma_block_y_size").get_to(compute->dma_block_y_size);
        j.at("x_min").get_to(compute->dma_x_min);
        j.at("x_max").get_to(compute->dma_x_max);
        j.at("read_x_stride").get_to(compute->read_x_stride);
        j.at("y_min").get_to(compute->dma_y_min);
        j.at("y_max").get_to(compute->dma_y_max);
        j.at("read_y_stride").get_to(compute->read_y_stride);
        j.at("copy_mode").get_to(compute->copy_mode);
        j.at("ker_round").get_to(compute->ker_round);
        //j.at("ker_on_board").get_to(compute->ker_on_board);
        //j.at("ker_repeat").get_to(compute->ker_repeat);
        //j.at("ker_repeat_last").get_to(compute->ker_repeat_last);
        j.at("ker_addr_s").get_to(compute->ker_addr_s);
        j.at("ker_addr_e").get_to(compute->ker_addr_e);
        j.at("output_num").get_to(compute->output_num);
        j.at("channel_out").get_to(compute->channel_out);
        j.at("output_channel").get_to(compute->output_channel);
        j.at("shift_num_fm").get_to(compute->shift_num_fm);
        j.at("shift_num_bias").get_to(compute->shift_num_bias);
        j.at("add_bias").get_to(compute->add_bias);
        j.at("add_temp").get_to(compute->add_temp);
        j.at("final_output").get_to(compute->final_output);
        j.at("output_block_y_size").get_to(compute->output_block_y_size);
        j.at("output_block_x_size").get_to(compute->output_block_x_size);
        if (j.contains("psum_write_addr_s")) {
            j.at("psum_write_addr_s").get_to(compute->psum_addr_s);
        }
        if (j.contains("add_zero")) {
            j.at("add_zero").get_to(compute->add_zero);
        }
        dev->RunCompute(compute);
    } else if (opcode == "barrier") {
        return 0;
    }
    return 1;
}

void run_lines(std::ifstream &inputStream, Device* dev, int layer_index, int line_cnt) {
    LOG_INFO << "\nLayer " << layer_index << "\n";
    int i = 0;
    for (std::string line; std::getline(inputStream, line); ) {
        if (run_one_line(line, dev) == 0) {
            break;
        }
        i++;
        if (i == line_cnt) break;
    }
}

void run_one_layer(std::ifstream &inputStream, Device* dev, int layer_index) {
    LOG_INFO << "\nLayer " << layer_index << "\n";
    if (layer_index == 2) {
        dev->debug = true;
        //dev->ipa_.debug = true;
        //std::cout << "(press enter to continue)"; std::cin.get();
    }//std::cout << "(press enter to continue)"; std::cin.get();
    int i = 0;
    for (std::string line; std::getline(inputStream, line); ) {
        if (run_one_line(line, dev) == 0) {
            break;
        }
        i++;
        //if (i == 53+44+49*22+2 + 47+49*23+2) break;
        //if (layer_index == 4 && i == 67) break;
    } //exit(1); 
    #ifdef BERT
    if (layer_index == 1) {  // .T
        dev->dram.SaveNumpy(234032, /*row*/512, /*col*/768, /*ofmap fraclen*/13, "ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 2) {
        dev->dram.SaveNumpy(246320, /*row*/512, /*col*/768, /*ofmap fraclen*/11, "ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 3) {
        dev->dram.SaveNumpy(258608, /*row*/512, /*col*/768, /*ofmap fraclen*/12, "ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 4) {
        dev->dram.SaveNumpy(270896, /*row*/512, /*col*/512*12, /*ofmap fraclen*//*14*/15, "ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 5) {
        dev->dram.SaveNumpy(369200, /*row*/512, /*col*/768, /*ofmap fraclen*/13, "ofmap_" + std::to_string(layer_index)); 
    }
    if (layer_index == 6) {
        dev->dram.SaveNumpy(381488, /*row*/512, /*col*/768, /*ofmap fraclen*/9, "ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 7) {
        dev->dram.SaveNumpy(393776, /*row*/512, /*col*/3072, /*ofmap fraclen*/11, "ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 8) {
        dev->dram.SaveNumpy(442928, /*row*/512, /*col*/768, /*ofmap fraclen*/11, "ofmap_" + std::to_string(layer_index));
        exit(1);
    }
    if (layer_index == 9) {
        //dev->dram.SaveNumpy(473696, /*row*/32, /*col*/768, /*ofmap fraclen*/11, "ofmap_" + std::to_string(layer_index));
        //exit(1);
    }
    if (layer_index == 80) {
        //dev->dram.SaveNumpy(86240, /*row*/512, /*col*/768, /*ofmap fraclen*/13, "ofmap_" + std::to_string(layer_index));
        //dev->dram.SaveNumpy(98528, /*row*/768, /*col*/512, /*ofmap fraclen*/11, "ofmap_" + std::to_string(layer_index));
        //dev->dram.SaveNumpy(110816, /*row*/512, /*col*/768, /*ofmap fraclen*/12, "ofmap_" + std::to_string(layer_index));
        //dev->dram.SaveNumpy(270896, /*row*/512, /*col*/512*12, /*ofmap fraclen*//*14*/15, "ofmap_" + std::to_string(layer_index));
        //dev->dram.SaveNumpy(369200, /*row*/512, /*col*/768, /*ofmap fraclen*/13, "ofmap_" + std::to_string(layer_index));
        //dev->dram.SaveNumpy(381488, /*row*/512, /*col*/768, /*ofmap fraclen*/9, "ofmap_" + std::to_string(layer_index));
        dev->dram.SaveNumpy(4891248, /*row*/512, /*col*/768, /*ofmap fraclen*/10, "ofmap_" + std::to_string(layer_index));
        exit(1);
    }
    #endif
    #ifdef VIT
    if (layer_index == 1) {
        dev->dram.SaveNumpy(288208, /*row*/224, /*col*/768, /*ofmap fraclen*/11, "v_ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 2) {
        dev->dram.SaveNumpy(293584, /*row*/224, /*col*/768, /*ofmap fraclen*/15, "v_ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 3) {
        dev->dram.SaveNumpy(298960, /*row*/224, /*col*/768, /*ofmap fraclen*/12, "v_ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 4) {
        dev->dram.SaveNumpy(304336, /*row*/224, /*col*/768, /*ofmap fraclen*/11, "v_ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 5) {
        dev->dram.SaveNumpy(309712, /*row*/224, /*col*/768, /*ofmap fraclen*/11, "v_ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 6) {
        dev->dram.SaveNumpy(315088, /*row*/224, /*col*/224*12, /*ofmap fraclen*/15, "v_ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 7) {
        dev->dram.SaveNumpy(333904, /*row*/224, /*col*/768, /*ofmap fraclen*/13, "v_ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 8) {
        dev->dram.SaveNumpy(339280, /*row*/224, /*col*/768, /*ofmap fraclen*/12, "v_ofmap_" + std::to_string(layer_index));
        dev->dram.SaveNumpy(371536, /*row*/224, /*col*/768, /*ofmap fraclen*/11, "v_ofmap_103");
    }
    if (layer_index == 9) {
        dev->dram.SaveNumpy(344656, /*row*/224, /*col*/3072, /*ofmap fraclen*/12, "v_ofmap_" + std::to_string(layer_index));
    }
    if (layer_index == 10) {
        dev->dram.SaveNumpy(366160, /*row*/224, /*col*/768, /*ofmap fraclen*/13, "v_ofmap_" + std::to_string(layer_index));
    }
    #endif
}

int get_layer_cnt(std::string filename) {
    std::ifstream inputFile(filename);
    int layer_cnt = 0;
    for (std::string line; std::getline(inputFile, line); ) {
        nlohmann::json j = nlohmann::json::parse(line);
        std::string opcode;
        j.at("opcode").get_to(opcode);
        if (opcode == "barrier") {
            layer_cnt++;
        }
    }
    inputFile.close();
    return layer_cnt;
}

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
    //LOG(INFO) << "info";
    //GLOG_v = 0;
    //VLOG(1) << "this is 1";
    
    Device* dev = new Device();
    std::string filename = argv[1];
    std::cout << "read from " << filename << "\n";
    int layer_cnt;
    if (argc == 2) {
        layer_cnt = get_layer_cnt(filename);
        std::cout << "#layer count = " << layer_cnt << "\n";
    } else {
        layer_cnt = std::atoi(argv[2]);
    }
    // initialize dram
    std::string dram_bin_filename = argv[3];
    dev->dram.base = dev->dram.Alloc(2 << 28);
    dev->dram.FromFile(0, dram_bin_filename);
    open_debug_file(dev);
    // run layer by layer
    std::ifstream inputFile(filename);
    //run_lines(inputFile, dev, 1, 2);
    //dev->dram.SaveNumpy(140464, /*row*/224, /*col*/768, /*ofmap fraclen*/11, "v_ofmap_1");
    for (int i = 0; i < layer_cnt; i++) {
        if (i == layer_cnt - 1) {
            dev->dump = true;
        }
        run_one_layer(inputFile, dev, i + 1);
    }
    inputFile.close();
    dev->CloseAllOfile();
    dev->os.close();
    return 0;
}