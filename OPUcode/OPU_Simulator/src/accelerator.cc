#include "./accelerator.h"
#include <glog/logging.h>
#include "./logging.h"
/*
 * Function wrapper for DDR load instruction, which tarverses
 * load options and perform memory transaction from DDR to SRAM
 * accordingly for feature map, weight, bias, instruction
 */
void debug_print_load_dram2buffer(int dram_addr_offset, int step_size, int step_count, int stride, std::string buffer_name, int buffer_bank_id) {
  LOG_INFO << "Load from DRAM["
            << "start addr:" << dram_addr_offset 
            << " step size:" << step_size
            << " x " << step_count 
            << " stride:" << stride
            << "] to " << buffer_name
            << "[" << buffer_bank_id <<"]\n";
}

void Device::LoadIns(int ddr_ins_addr) {
  debug_print_load_dram2buffer(
    ddr_ins_addr, INS_BUFFER_DEPTH * INS_BUFFER_WIDTH / DRAM_BANDWIDTH,
    1, 1,
    "inst_buffer", ins_ram_id
  ); 
  Ins_ram_t* ins_ram = ins_ram_vec_[ins_ram_id];
  ins_ram_id = (ins_ram_id + 1) % ins_ram_vec_.size();
  uint32_t *data = reinterpret_cast<uint32_t*>(dram.GetAddr(ddr_ins_addr));
  uint32_t *ins_tmp = new uint32_t[INS_BUFFER_DEPTH];
  for (int i = 0; i < INS_BUFFER_DEPTH; i += DRAM_BANDWIDTH / INS_BUFFER_WIDTH) {
    for (int j = 0; j < DRAM_BANDWIDTH / INS_BUFFER_WIDTH; j++) {
      ins_tmp[i + j] = data[i + DRAM_BANDWIDTH / INS_BUFFER_WIDTH - 1 - j];
    }
  }
  MemOp mem = MemOp(0, INS_BUFFER_DEPTH, 0);
  ins_ram->Load<INS_BUFFER_WIDTH>(mem, ins_tmp);
  delete [] ins_tmp;
}

void Device::RunLoad(OPUDDRLDInsn* load) {
  //std::cout << "*************DDR LD**************\n";
  load_single = load->ddr_load_single;
  for (auto mem_id : load->ddr_load_type) {
    //LOG_INFO << mem_id << "\n";
    if (mem_id == OPU_MEM_ID_FM) {  // load feature map
      debug_print_load_dram2buffer(
        load->ddr_fm_addr_ini, load->ddr_load_block_x_size, 
        load->ddr_load_block_y_size, load->fm_in_x_size,
        "ifmap_buffer", fm_ram_id
      );
      Fm_ram_t* fm_ram = fm_ram_vec_[fm_ram_id];
      fm_ram_id = (fm_ram_id + 1) % fm_ram_vec_.size();
      for (int i = 0; i < load->ddr_load_block_y_size; i++) {
        size_t src_addr = load->ddr_fm_addr_ini + i * load->fm_in_x_size;
        size_t dst_addr = i * load->ddr_load_block_x_size;
        MemOp mem = MemOp(
            src_addr,
            load->ddr_load_block_x_size,
            dst_addr);
        fm_ram->Load<DRAM_BANDWIDTH>(mem, dram.GetAddr(mem.src_addr));
        /*if (debug && load->ddr_fm_addr_ini == 123616) {
          std::cout << "$$" << src_addr << " -> " << dst_addr << "\n";
          for (int u = dst_addr; u < dst_addr + 32; u++) {
            std::cout << "<<<<<<" << u << ">>>>>>>>\n";
            std::vector<sim_data_type> t = fm_ram->AsVec(u, DRAM_BANDWIDTH/FMAP_DATA_WIDTH, FMAP_DATA_WIDTH);
            for (auto &x : t) std::cout << (double)x / (1 << 14) << " ";std::cout << "\n";
          }
        }*/
      }//if(debug && load->ddr_fm_addr_ini == 123616)exit(1);
    } else if (mem_id == OPU_MEM_ID_WGT) {  // load weight
      debug_print_load_dram2buffer(
        load->ddr_ker_addr_ini, load->ddr_ker_read_num,
        1, 1,
        "wgt_buffer", wgt_ram_id
      );
      Wgt_ram_t* wgt_ram = wgt_ram_vec_[wgt_ram_id];
      wgt_ram_id = (wgt_ram_id + 1) % wgt_ram_vec_.size();
      MemOp mem = MemOp(
            load->ddr_ker_addr_ini,
            load->ddr_ker_read_num,
            0);
      wgt_ram->Load<DRAM_BANDWIDTH>(mem, dram.GetAddr(mem.src_addr));
      /*if (debug) {
        std::cout << "$$" << mem.src_addr << " -> " << mem.dst_addr << "\n";
        for (int u = mem.src_addr; u < mem.src_addr + 32; u++) {
          std::cout << "<<<<<<" << u << ">>>>>>>>\n";
          fmap_data_type *data = reinterpret_cast<fmap_data_type*>(dram.GetAddr(u));
          for (int i = 0; i < 32; i++) {
            fmap_data_type value = REVERSE_BYTES16(data[i]);
            std::cout << (double)value / (1 << 13) << " ";
          }std::cout << "\n";
        }
        exit(1);
      }*/
    } else if (mem_id == OPU_MEM_ID_BIAS) {  // load bias
      debug_print_load_dram2buffer(
        load->ddr_bias_addr_ini, load->ddr_bias_read_num,
        1, 1,
        "bias_buffer", bias_ram_id
      );
      Bias_ram_t* bias_ram = bias_ram_vec_[bias_ram_id];
      bias_ram_id = (bias_ram_id + 1) % bias_ram_vec_.size();
      MemOp mem = MemOp(
            load->ddr_bias_addr_ini,
            load->ddr_bias_read_num,
            0);
      bias_ram->Load<DRAM_BANDWIDTH>(mem, dram.GetAddr(mem.src_addr));
      /*if (debug) {
        std::cout << "$$" << mem.src_addr << " -> " << mem.dst_addr << "\n";
        for (int u = mem.src_addr; u < mem.src_addr + 2; u++) {
          std::cout << "<<<<<<" << u << ">>>>>>>>\n";
          psum_data_type *data = reinterpret_cast<psum_data_type*>(dram.GetAddr(u));
          for (int i = 0; i < 16; i++) 
            std::cout << (double)data[i] / (1 << 13) << " ";std::cout << "\n";
        }
        exit(1);
      }*/
    } else if (mem_id == OPU_MEM_ID_RESIDUAL) {  // load feature map for residual add

    } else if (mem_id == OPU_MEM_ID_INS) {  // load instruction
      LoadIns(load->ddr_ins_addr_ini);
    } else if (mem_id == OPU_MEM_ID_PARAM) {  // load param
      debug_print_load_dram2buffer(
        load->ddr_param_addr_ini, load->ddr_param_read_num,
        1, 1,
        "param_buffer", load->param_ram_id
      );
      MemOp mem = MemOp(
            load->ddr_param_addr_ini,
            load->ddr_param_read_num,
            nvm_.GetBankAddr(load->param_ram_id, 0));
      nvm_.vregs_.Load<DRAM_BANDWIDTH>(mem, dram.GetAddr(mem.src_addr));
    } else if (mem_id == OPU_MEM_ID_PSUM) {  // load dram -> psum
      debug_print_load_dram2buffer(
        load->ddr_fm_addr_ini, load->ddr_load_block_x_size, 
        load->ddr_load_block_y_size, load->fm_in_x_size,
        "psum_buffer", load->psum_addr_s
      );
      Psum_t* psum = &tmp_buf_;
      for (int i = 0; i < load->ddr_load_block_y_size; i++) {
        size_t src_addr = load->ddr_fm_addr_ini + i * load->fm_in_x_size;
        size_t dst_addr = load->psum_addr_s + i * load->ddr_load_block_x_size;
        for (size_t j = 0; j < load->ddr_load_block_x_size; j++) {
          std::vector<fmap_data_type> data(DRAM_BANDWIDTH/FMAP_DATA_WIDTH, 0);
          std::memcpy(&data[0], dram.GetAddr(src_addr + j), DRAM_BANDWIDTH/8);  // bytes count
          std::vector<psum_data_type> pdata;
          for (int k = 0; k < DRAM_BANDWIDTH/FMAP_DATA_WIDTH; k++) {
            fmap_data_type x = data[k];
            if (FMAP_DATA_WIDTH == 16) x = REVERSE_BYTES16(x);
            psum_data_type value = static_cast<psum_data_type>(x);
            //std::cout << (double)value / (1<<11) << " ";
            pdata.push_back(REVERSE_BYTES(value));
          }//std::cout << "\n";
          void *dst = psum->BeginPtr(dst_addr + j);
          std::memcpy(dst, &pdata[0], DRAM_BANDWIDTH/FMAP_DATA_WIDTH*PSUM_DATA_WIDTH/8); 
          /*std::cout << "psum <<<" << dst_addr + j << ">>>\n";
          psum_data_type* pd = reinterpret_cast<psum_data_type*>(psum->BeginPtr(dst_addr + j));
          for (int k = 0; k < DRAM_BANDWIDTH/FMAP_DATA_WIDTH; k++) {
            int64_t value = REVERSE_BYTES(pd[k]);
            std::cout << (double)value / (1<<11) << " ";
          } std::cout << "\n";*/
        }
      }
    } else {
      LOG(ERROR) << "[ERROR] unknown ddr load type:" << mem_id << "\n";
    }
  }
  
  // cycle count
  int64_t ddr_init_lat = 20;
  int64_t ddr_burst_length = 8;
  profiler.collect(0);
  for (auto mem_id : load->ddr_load_type) {
    if (mem_id == OPU_MEM_ID_FM) {
      for (int i = 0; i < load->ddr_load_block_y_size; i++) {
        profiler.incrementLast(
            static_cast<int64_t>(
            std::ceil(static_cast<double>(load->ddr_load_block_x_size)
            / static_cast<double>(ddr_burst_length)))
            + ddr_init_lat);
      }
    } else if (mem_id == OPU_MEM_ID_WGT) {
      profiler.incrementLast(
        static_cast<int64_t>(
        std::ceil(static_cast<double>(load->ddr_ker_read_num)
        / static_cast<double>(ddr_burst_length)))
        + ddr_init_lat);
    } else if (mem_id == OPU_MEM_ID_BIAS) {
      profiler.incrementLast(
        static_cast<int64_t>(
        std::ceil(static_cast<double>(load->ddr_bias_read_num)
        / static_cast<double>(ddr_burst_length)))
        + ddr_init_lat);
    }
  }
}

/*
 * Function wrapper for DDR store instruction, which applies post CONV processings,
 * including activation, pooling, residual addition, padding, (upsampling) in the
 * order dynamically spefcified by instructions
 */
void Device::RunStore(OPUDDRSTInsn* store) {
  std::stringstream ss;
  ss << "ordering: " << store->ddr_save_pos << " ";
  if (store->ddr_save_pos > 4) ss << "(row access pattern) ";
  else if (store->ddr_save_pos == 4) ss << "(block access pattern for transpose) ";
  else ss << "(pooling access pattern) ";
  if (store->residual) ss << "residual ";
  if (store->activation) ss << "activation(" << store->activation_type << ") ";
  if (store->pooling) ss << "pooling(" << store->pooling_type << ") ";
  if (store->padding) ss << "padding(" << store->padding_size << ") ";
  LOG_INFO << "Post Ops: " << ss.str() << "\n";
  LOG_INFO << "Store"
            //<< " (x)" << store->ddr_save_block_x_size
            //<< " x (y)" << store->ddr_save_block_y_size
            << " to DRAM start addr:" << store->fm_output_addr_ini 
            << " step_size:" << store->ddr_save_block_x_size
            << " step:" << store->ddr_save_block_y_size
            << " stride:" << store->fm_out_x_size
            << " count:" << store->ddr_save_fm_num <<"\n";
  LOG_DEBUG << "*************DDR ST**************\n";
  LOG_DEBUG << "ddr_save_pos = " << store->ddr_save_pos << "\n";
  LOG_DEBUG << "residual = " << store->residual << "\n";
  if (store->residual)
    LOG_DEBUG << "fm[" << ((compute_ins_->fm_ram_id + fm_ram_vec_.size() - 1)
    % fm_ram_vec_.size()) << "]\n";
  LOG_DEBUG << "activation_enable = " << store->activation
            << " type = " << store->activation_type << "\n";
  LOG_DEBUG << "pooling_enable = " << store->pooling
            << " type = " << store->pooling_type << "\n";
  LOG_DEBUG << "pooling_x_size = " << store->pooling_x_size
            << "(" << store->pooling_x_stride << ")\n";
  LOG_DEBUG << "pooling_y_size = " << store->pooling_y_size
            << "(" << store->pooling_y_stride << ")\n";
  LOG_DEBUG << "ST to ddr addr ini = " << store->fm_output_addr_ini << "\n";
  LOG_DEBUG << "ddr_save_fm_num = " << store->ddr_save_fm_num << "\n";
  LOG_DEBUG << "ddr_save_block_x_size = "
            << store->ddr_save_block_x_size << "\n";
  LOG_DEBUG << "ddr_save_block_y_size = "
            << store->ddr_save_block_y_size << "\n";
  LOG_DEBUG << "fm_out_x_size = " << store->fm_out_x_size << "\n";
  LOG_DEBUG << "fm_out_y_size = " << store->fm_out_y_size << "\n";
  LOG_DEBUG << "padding = " << store->padding << "\n";
  LOG_DEBUG << "padding_size = " << store->padding_size << "\n";
  LOG_DEBUG << "upsampling = " << store->upsample_output << "\n";
  LOG_DEBUG << "channel_out = " << store->channel_out << "\n";
  if (skip_exec) return;
  
  if (store->padding == 1) {
    // Padding
    RunPadding(store);
  } else {
    // Post-processing
    RunPostProcess(store);
  }

  // Debug
  if (store->padding == 1) {
    std::ofstream os_fm("ofm.txt");
    for (int i = 0; i < store->fm_out_x_size * store->fm_out_y_size * std::ceil((float)store->channel_out / (DRAM_BANDWIDTH/FMAP_DATA_WIDTH)); i++) {
      int addr = store->fm_output_addr_ini + i;
      std::vector<fmap_data_type> trunc(DRAM_BANDWIDTH/FMAP_DATA_WIDTH);
      void* data = dram.GetAddr(addr);
      std::memcpy(&trunc[0], data, DRAM_BANDWIDTH/FMAP_DATA_WIDTH);
      std::vector<sim_data_type> res;
      for (auto x : trunc) {
        res.push_back(static_cast<sim_data_type>(x));
      }
      writeOut<sim_data_type, 2>(os_fm, res, true);
    }
    os_fm.close();
  }
}

/*
 * Padding output feature map with zeros
 */
void Device::RunPadding(OPUDDRSTInsn* store) {
  LOG_INFO << "padding " << store->padding_size;
  std::vector<fmap_data_type> res8(DRAM_BANDWIDTH/FMAP_DATA_WIDTH, 0);
  int padding_size_l = store->padding_size;
  int padding_size_r = store->padding_size;
  int padding_size_u = store->padding_size;
  int padding_size_d = store->padding_size;
  int pad_cnt = 0;
  // Data layout :
  //   (row major) fm_x*fm_y*channel[63:0] - fm_x*fm_y*channel[127:64]
  for (int p = 0; p < store->channel_out; p += DRAM_BANDWIDTH/FMAP_DATA_WIDTH) {
    int addr = store->fm_output_addr_ini
        + p * store->fm_out_y_size * store->fm_out_x_size;
    // Top
    for (int i = 0; i < padding_size_u; i++) {
      for (int j = 0; j < store->fm_out_x_size; j++) {
        dram.Write(addr++, &res8[0], DRAM_BANDWIDTH/FMAP_DATA_WIDTH);
        pad_cnt++;
      }
    }
    CHECK_EQ(addr, store->fm_output_addr_ini
        + p * store->fm_out_y_size * store->fm_out_x_size
        + padding_size_u * store->fm_out_x_size);
    // Two sides
    for (int i = 0;
        i < store->fm_out_y_size - padding_size_u - padding_size_d; i++) {
      addr = store->fm_output_addr_ini
        + p * store->fm_out_y_size * store->fm_out_x_size
        + (padding_size_u + i) * store->fm_out_x_size;
      for (int j = 0; j < padding_size_l; j++) {
        dram.Write(addr + j, &res8[0], DRAM_BANDWIDTH/FMAP_DATA_WIDTH);
        pad_cnt++;
      }
      for (int j = 0; j < padding_size_r; j++) {
        dram.Write(addr + store->fm_out_x_size - padding_size_r + j,
            &res8[0], DRAM_BANDWIDTH/FMAP_DATA_WIDTH);
        pad_cnt++;
      }
    }
    // Bottom
    addr = store->fm_output_addr_ini
        + p * store->fm_out_y_size * store->fm_out_x_size
        + (store->fm_out_y_size - padding_size_d) * store->fm_out_x_size;
    for (int i = 0; i < padding_size_d; i++) {
      for (int j = 0; j < store->fm_out_x_size; j++) {
        dram.Write(addr++, &res8[0], DRAM_BANDWIDTH/FMAP_DATA_WIDTH);
        pad_cnt++;
      }
    }
    CHECK_EQ(addr, store->fm_output_addr_ini
        + p * store->fm_out_y_size * store->fm_out_x_size
        + store->fm_out_y_size * store->fm_out_x_size);
  }
  // Check total padding count
  CHECK_EQ(pad_cnt, store->ddr_save_fm_num);
  profiler.collect(pad_cnt);
}

/*
 * Post-processing
 */
void Device::RunPostProcess(OPUDDRSTInsn* store) {
  if (store->ddr_save_pos >=0 && store->ddr_save_pos <= 3) {  // element-wise, fine-grained pipeline
    RunPostProcess_legacy(store);
  } else if (store->ddr_save_pos >= 4 && store->ddr_save_pos <= 6) {
    RunPostProcess_transpose(store);
  } else {  // tiled, coarse-grained pipeline
    RunPostProcess_nvm(store);
  }
}

void Device::RunPostProcess_transpose(OPUDDRSTInsn* store) {
  int psum_addr_s = store->psum_addr_s;  // 0
  int psum_read_stride = store->ddr_save_block_x_size;
  int rows_in_psum = store->ddr_save_block_x_size;
  int dram_write_offset = store->fm_output_addr_ini;
  int dram_write_stride = store->fm_out_x_size;
  int row_block = 0;  // #32x32blocks in one tile = 2 in 64(row)x32(col) tile
  int transpose_block_size = 32;

  for (int i = 0; i < store->ddr_save_block_y_size; i++) {
    //std::cout << "=============== " << i << "================\n";
    while (row_block * transpose_block_size < rows_in_psum) {
      // read a 32x32 block from psum
      std::vector<std::vector<sim_data_type>> block;
      for (int j = 0; j < transpose_block_size; j++) {
        int addr = psum_addr_s + i * psum_read_stride + row_block * transpose_block_size + j;
        std::vector<sim_data_type> data = tmp_buf_.AsVec(addr, DRAM_BANDWIDTH/FMAP_DATA_WIDTH, PSUM_DATA_WIDTH);
        block.push_back(data);
      }
      // write to dram
      for (int j = 0; j < transpose_block_size; j++) {
        std::vector<fmap_data_type> transposed;
        for (int p = 0; p < transpose_block_size; p++) {
          if (store->ddr_save_pos == 4) {  // transpose
            transposed.push_back(static_cast<fmap_data_type>(block[p][j]));
          } else {
            transposed.push_back(static_cast<fmap_data_type>(block[j][p]));
          }
        }
        //for (auto &x : transposed) std::cout << (double)x / (1<<11) << " ";std::cout << "\n"; 
        std::vector<char> bytes;
        for (int ii = 0; ii < DRAM_BANDWIDTH/FMAP_DATA_WIDTH; ii++) {
          if (FMAP_DATA_WIDTH == 8) {
            bytes.push_back(static_cast<char>(transposed[ii]));
          } else if (FMAP_DATA_WIDTH == 16) {
            sim_data_type val = transposed[ii];
            sim_data_type base = 1 << 8;
            bytes.push_back((sim_data_type)floor((double)val / base));  // high 16bit
            bytes.push_back(val % base);  // low 16bit
          }
        } 
        int ddr_addr; 
        if (store->ddr_save_pos == 4 || store->ddr_save_pos == 5) {  // column-major
          ddr_addr = dram_write_offset + i * dram_write_stride + row_block * transpose_block_size + j;
        } else if (store->ddr_save_pos == 6) {  // row-major
          ddr_addr = dram_write_offset + i * transpose_block_size + 
            row_block * dram_write_stride + j;
        }
        //std::cout << "<<<<" << ddr_addr << ">>>>\n";
        //for (auto &x : transposed) std::cout << (double)x / (1<<13) << " ";std::cout << "\n";
        dram.Write(ddr_addr, &bytes[0], DRAM_BANDWIDTH/FMAP_DATA_WIDTH*(FMAP_DATA_WIDTH/8)); 
      }
      row_block++;
    }
    row_block = 0;
  }//exit(1);
}

void Device::RunPostProcess_nvm(OPUDDRSTInsn* store) {
  int64_t initiate_cnt = 0;
  // psum read address gen: [(psum_read_s+i*psum_read_stride) +: store->ddr_save_block_x_size] 
  // use store->channel_out to control i
  int psum_addr_s = 0;//store->psum_addr_s;
  int psum_read_stride = store->psum_read_stride;
  int row_in_psum = store->channel_out;  // 64
  int nvm_bank_depth_per_row = store->ddr_save_block_y_size;// store->fm_out_x_size / (DRAM_BANDWIDTH/FMAP_DATA_WIDTH);  // 768 / 32 = 24
  int nvm_banks = NVM_t::banks;
  int dram_write_offset = store->fm_output_addr_ini;
  int dram_write_stride = DRAM_BANDWIDTH/FMAP_DATA_WIDTH;

  int residue_read_stride = 32;

  if (store->residual) {
    int rid = (compute_ins_->fm_ram_id + fm_ram_vec_.size() - 1) % fm_ram_vec_.size();
    LOG_INFO << "residue from ifmap buffer[" << rid << "]\n";
  }

  // row pipeline
  std::vector<sim_data_type> res;
  for (int p = 0; p < row_in_psum; p++) {
    //std::cout << "================= row " << p << " =================\n";
    res.clear();
    // read one row from psum to nvm reg
    for (int k = 0; k < nvm_bank_depth_per_row; k++) {
      int psum_read_addr = psum_addr_s + p + k * row_in_psum;
      //nvm_.regs_.Load<PSUM_DATA_WIDTH*DRAM_BANDWIDTH/FMAP_DATA_WIDTH>(
      //  MemOp(psum_read_addr, 1, nvm_.GetBankAddr(/*bank id*/0, /*offset*/k)),
      //  tmp_buf_.BeginPtr(psum_addr_s)
      //);
      std::vector<sim_data_type> data = tmp_buf_.AsVec(psum_read_addr, DRAM_BANDWIDTH/FMAP_DATA_WIDTH, PSUM_DATA_WIDTH);
      for (auto &x : data) res.push_back(x);
      /*if (debug) {
        std::cout << "<<" << psum_read_addr << ">>\n";
        for (auto &x : res) std::cout << (double)x / (1<<11) << " ";
        std::cout << "\n";
      }*/
    }
    // post ops
    if (store->residual) {
      // TODO : 1. fetch from which buffer for residue 2. data fetch pattern 3. residue ofmap alignment
      std::vector<sim_data_type> residue;
      int rid = (compute_ins_->fm_ram_id + fm_ram_vec_.size() - 1) % fm_ram_vec_.size();
      // LOG_INFO << "residue from ifmap buffer[" << rid << "]\n";
      for (int k = 0; k < nvm_bank_depth_per_row; k++) {
        //int addr = p / residue_read_stride * store->fm_out_x_size + p % residue_read_stride + k * residue_read_stride;
        int addr = psum_addr_s + p + k * row_in_psum;
        std::vector<sim_data_type> data = fm_ram_vec_[rid]->AsVec(addr, DRAM_BANDWIDTH/FMAP_DATA_WIDTH, FMAP_DATA_WIDTH);
        for (auto &x : data) // frontend match fl to the smaller one
          residue.push_back(x);  // no saturation for input, otherwise lose too much precision 
      }
      for (int k = 0; k < nvm_bank_depth_per_row; k++) {
        for (int kk = 0; kk < 32; kk++) {
          res[k * 32 + kk] += residue[k * 32 + kk];
        }
      }
      Saturate(res, /*output data width*/FMAP_DATA_WIDTH);
    }
    // activation
    if (store->activation) {
      if (store->activation_type == 5) {
        nvm_.input_fraclen = store->nvm_input_fraclen;
        nvm_.output_fraclen = store->nvm_output_fraclen;
        nvm_.Run<sim_data_type>(res, NonlinearOpcode::FloatGeLU);
      }
    }

    if (store->ddr_save_pos == 8) {  // divide - softmax
      // divide by 8
      for (auto &x : res) x = x >> 3;
      // softmax
      //for (auto &x : res) std::cout << (double)x / (1 << 8) << " ";std::cout << "\n\n";
      nvm_.input_fraclen = store->nvm_input_fraclen; // 8
      nvm_.output_fraclen = store->nvm_output_fraclen;  // 15
      nvm_.Run<sim_data_type>(res, NonlinearOpcode::FloatSoftmax);
      //nvm_.Run<sim_data_type>(res, NonlinearOpcode::Softmax);
      //for (auto &x : res) std::cout << (double)x / (1 << 15) << " ";std::cout << "\n";exit(1); 
    } else if (store->ddr_save_pos == 7) {  // layernorm
      nvm_.input_fraclen = store->nvm_input_fraclen; // 11;  // psum
      nvm_.output_fraclen = store->nvm_output_fraclen;  // 9;  // ofmap
      nvm_.beta_fraclen = store->nvm_beta_fraclen;
      nvm_.gamma_fraclen = store->nvm_gamma_fraclen;
      nvm_.rows = nvm_bank_depth_per_row;
      nvm_.Run<sim_data_type>(res, NonlinearOpcode::FloatLayerNorm);
      //nvm_.Run<sim_data_type>(res, NonlinearOpcode::LayerNorm);
    }
    
    // store
    std::vector<fmap_data_type> res_truncated(DRAM_BANDWIDTH/FMAP_DATA_WIDTH);
    for (int k = 0; k < nvm_bank_depth_per_row; k++) {
      //std::vector<sim_data_type> data = nvm_.vregs_.AsVec(nvm_.GetBankAddr(0, k), DRAM_BANDWIDTH/FMAP_DATA_WIDTH, FMAP_DATA_WIDTH);
      for (int i = 0; i < DRAM_BANDWIDTH/FMAP_DATA_WIDTH; i++) {
        res_truncated[i] = static_cast<fmap_data_type>(res[k * DRAM_BANDWIDTH/FMAP_DATA_WIDTH + i]);
      }
      //int ddr_addr = dram_write_offset + p % 32 + k * dram_write_stride + (p / 32) * store->fm_out_x_size;
      int ddr_addr = dram_write_offset + p + k * store->fm_out_x_size;
      std::vector<char> bytes;
      for (int i = 0; i < DRAM_BANDWIDTH/FMAP_DATA_WIDTH; i++) {
        if (FMAP_DATA_WIDTH == 8) {
          bytes.push_back(static_cast<char>(res_truncated[i]));
        } else if (FMAP_DATA_WIDTH == 16) {
          sim_data_type val = res_truncated[i];
          sim_data_type base = 1 << 8;
          bytes.push_back((sim_data_type)floor((double)val / base));  // high 16bit
          bytes.push_back(val % base);  // low 16bit
        }
      }   
      dram.Write(ddr_addr, &bytes[0], DRAM_BANDWIDTH/FMAP_DATA_WIDTH*(FMAP_DATA_WIDTH/8));  
      /*if (debug) {
        std::cout << "<<<<" << ddr_addr << ">>>>\n";
        int16_t* t = reinterpret_cast<int16_t*>(dram.GetAddr(ddr_addr));
        for (int i = 0; i < DRAM_BANDWIDTH/FMAP_DATA_WIDTH; i++) {
          int16_t value = t[i];
          if (FMAP_DATA_WIDTH == 16) {
            value = ((value & 0xFF) << 8) | ((value & 0xFF00) >> 8);
          }
          std::cout << (double)value / ((int64_t)1 << 11) << " ";
        }std::cout << "\n";
      }*/
    }
  }//if (debug) {std::cout << "(press enter to continue)"; std::cin.get();}

  int64_t ii = 1;
  int64_t epilogue = 20;
  int64_t latency = initiate_cnt * ii + epilogue;
  profiler.collect(latency);
}

/*
 * Post-processing according to pooling's memory access pattern
 */
void Device::RunPostProcess_legacy(OPUDDRSTInsn* store) {
  int64_t initiate_cnt = 0;
  // psum read address gen: [(psum_read_s+i*psum_read_stride) +: store->ddr_save_block_x_size] 
  int psum_addr_s = store->psum_addr_s;  // 0
  int psum_read_stride = store->ddr_save_block_x_size;
  std::vector<sim_data_type> res(DRAM_BANDWIDTH/FMAP_DATA_WIDTH);
  // fetch ofm buffer according to pooling pattern
  for (int i = 0;
        i < store->ddr_save_block_y_size - store->pooling_y_size + 1;
        i += store->pooling_y_stride) {
    for (int j = 0;
        j < store->ddr_save_block_x_size - store->pooling_x_size + 1;
        j += store->pooling_x_stride) {
      for (int y = 0; y < store->pooling_y_size; y++) {
        for (int x = 0; x < store->pooling_x_size; x++) {
          initiate_cnt++;
          int addr = psum_addr_s + (i+y) * psum_read_stride + j + x;
          std::vector<sim_data_type> temp = tmp_buf_.AsVec(addr, DRAM_BANDWIDTH/FMAP_DATA_WIDTH, PSUM_DATA_WIDTH);
          bool elew_st = x == 0 && y == 0;
          bool elew_ed = x == store->pooling_x_size - 1 && y == store->pooling_y_size - 1;
          int window_size = store->pooling_x_size * store->pooling_y_size;
          if (store->ddr_save_pos == 3) {
            Activation(temp, store->activation_type);
            Pooling(res, temp, store->pooling_type, elew_st, elew_ed, window_size);
          } else if (store->ddr_save_pos == 2) {
            Activation(temp, store->activation_type);
            Pooling(res, temp, store->pooling_type, elew_st, elew_ed, window_size);
            ResidualAdd(res, store->residual, addr);
          } else if (store->ddr_save_pos == 1) {
            Activation(temp, store->activation_type);
            ResidualAdd(temp, store->residual, addr);
            Pooling(res, temp, store->pooling_type, elew_st, elew_ed, window_size);
          } else {
            ResidualAdd(temp, store->residual, addr);
            Activation(temp, store->activation_type);
            Pooling(res, temp, store->pooling_type, elew_st, elew_ed, window_size);
          }
        }
      }
      std::vector<fmap_data_type> res_truncated;
      for (auto x : res) {
        res_truncated.push_back(static_cast<fmap_data_type>(TRUNCATE(x)));
      }
      // Upsampling
      // TODO : scale currently fixed as 2
      int upsampling_scale = store->upsample_output? 2 : 1;
      for (int y = 0; y < upsampling_scale; y++) {
        for (int x = 0; x < upsampling_scale; x++) {
        #ifdef DEBUG_POST_OUT
          writeOut<sim_data_type, FMAP_DATA_WIDTH/4>(os, res, dump);
        #endif
          // Write results to DDR
          int coord_y = i / store->pooling_y_stride * upsampling_scale + y;
          int coord_x = j / store->pooling_x_stride * upsampling_scale + x;
          int ddr_addr = store->fm_output_addr_ini +
                coord_y * store->fm_out_x_size +
                coord_x;
          dram.Write(ddr_addr, &res_truncated[0], DRAM_BANDWIDTH/FMAP_DATA_WIDTH*(FMAP_DATA_WIDTH/8));  // write in bytes
        }
      }
    }
  }
  
  int64_t ii = 1;
  int64_t epilogue = 20;
  int64_t latency = initiate_cnt * ii + epilogue;
  profiler.collect(latency);//epilogue);
  /*if (reg_.dw_flag)
    profiler.collect(epilogue);
  else
    profiler.collect(latency);*/
}

/*
 * Residual Add
 */
void Device::ResidualAdd(std::vector<sim_data_type>& data, bool enable, int addr) {
  if (enable) {
    // Use the fm ram other than the one for compute
    // TODO : uniform buffer arbitration!
    int rid = (compute_ins_->fm_ram_id + fm_ram_vec_.size() - 1)
        % fm_ram_vec_.size();
    std::vector<sim_data_type> residue =
        fm_ram_vec_[rid]->AsVec(addr, DRAM_BANDWIDTH, FMAP_DATA_WIDTH);
    elw_unit_.Run(data, residue, ElwOpcode::Add);
    Saturate(data, /*output data width*/FMAP_DATA_WIDTH);
  } 
}

/*
 * Activation
 */
void Device::Activation(std::vector<sim_data_type>& data, int type) {
  ActivationType opcode;
  if (type == NO_RELU_CODE || reg_.activation == 0) {
    opcode = ActivationType::BypassActivation;
  } else if (type == RELU_CODE) {  // relu
    opcode = ActivationType::ReLU;
  } else if (type == LEAKY_RELU_CODE) {  // leaky_relu: *0.125 == >>3 and round
    opcode = ActivationType::pReLU;
  } else {
    opcode = ActivationType::BypassActivation;
  }
  activation_unit_.Run(data, opcode);
}

/*
 * Pooling
 */
void Device::Pooling
  (std::vector<sim_data_type>& data_o, std::vector<sim_data_type> data_i, int type, bool st, bool ed, int window_size) {
  auto it = data_o.begin();
  auto ie = data_i.begin();
  if (type == 0 || reg_.pooling == 0) {
    data_o.clear();
    data_o.insert(data_o.begin(), data_i.begin(), data_i.end());
  } else if (type == 1) {  // max
    while (it != data_o.end()) {
      if (st || *ie > *it) {
        *it = *ie;
      }
      it++;
      ie++;
    }
  } else if (type == 2) {  // avg
    while (it != data_o.end()) {
      if (st) {
        *it = *ie;
      } else {
        *it += *ie;
      }
      it++;
      ie++;
    }
    if (ed) {
      for (auto& item : data_o) {
        item /= window_size;  // precision issue?
      }
    }
  } else {
    data_o.clear();
    data_o.insert(data_o.begin(), data_i.begin(), data_i.end());
  }
}

/*
 * Function wrapper for inner product compuation
 */
void Device::RunCompute(OPUComputeInsn* compute) {
  std::stringstream ss;
  ss << "Inner-product " << "x[" << compute->fm_ram_id << "][" << compute->input_addr_s << ":]";
  ss << "(" << compute->dma_y_min << ":" << compute->dma_y_max << ")";
  if (compute->read_y_stride > 1) ss << "s" << compute->read_y_stride;
  ss << "(" << compute->dma_x_min << ":" << compute->dma_x_max << ")";
  if (compute->read_x_stride > 1) ss << "s" << compute->read_x_stride;
  ss << "(multicast " << (8 << compute->copy_mode) << ") " << "x ";
  ss << "w[" << compute-> wgt_ram_id << "][" << compute->ker_addr_s << ":" << compute->ker_addr_e << "] + ";
  if (compute->add_zero) ss << "0 ";
  if (compute->add_bias) ss << "bias[" << compute->bias_ram_id << "] ";
  if (compute->add_temp) ss << "psum[" << compute->psum_addr_s << ":] ";
  if (compute->final_output) ss << "-> store ";
  else ss << "-> psum[" << compute->psum_addr_s << ":] ";
  LOG_INFO << ss.str();

  LOG_DEBUG << "*************Compute**************\n";
  Fm_ram_t* fm_ram = fm_ram_vec_[compute->fm_ram_id];
  LOG_DEBUG << "Fetch from fm[" << compute->fm_ram_id <<"] addr_s = " << compute->input_addr_s << "\n";
  Wgt_ram_t* wgt_ram = wgt_ram_vec_[compute->wgt_ram_id];
  LOG_DEBUG << "Fetch from wgt[" << compute->wgt_ram_id <<"] addr_s = " << compute->ker_addr_s << "\n";
  Bias_ram_t* bias_ram = bias_ram_vec_[compute->bias_ram_id];
  LOG_DEBUG << "Fetch from bias[" << compute->bias_ram_id <<"]\n";
  LOG_DEBUG << "fm y[" << compute->dma_y_min << ":" << compute->dma_y_max << "]"
            << "x[" << compute->dma_x_min << ":" << compute->dma_x_max << "]\n";
  LOG_DEBUG << "final_output = " << compute->final_output << "\n";
  LOG_DEBUG << "psum_addr_s = " << compute->psum_addr_s << "\n";
  LOG_DEBUG << "ker_round = " << compute->ker_round << "\n";
  LOG_DEBUG << "shift_fm = " << compute->shift_num_fm << ", " << "shift_bias = " << compute->shift_num_bias << "\n";
  LOG_DEBUG << "dma_block_x_size = " << compute->dma_block_x_size << "\n";
  LOG_DEBUG << "copy_mode = " << compute->copy_mode << ", " << "output_num = " << compute->output_num << "\n";
  LOG_DEBUG << "output_channel(tile) = " << compute->output_channel << "\n";
  LOG_DEBUG << "stride = " << compute->read_y_stride << ", " << compute->read_x_stride << "\n";
  LOG_DEBUG << "add_bias = " << compute->add_bias << ", " << "add_temp = " << compute->add_temp << "\n";
  if (compute->dw_flag) {
    LOG_DEBUG << "[DW]\n";
    RunComputeDW(compute);
    return;
  }
  
  if (compute->type == 0) {
    LOG_DEBUG << "[FC]\n";
    RunComputeFC(compute);
    return;
  }
  
  // Instrumentation
  int64_t initiate_cnt = 0;
  // Control flags
  compute_finish = compute->final_output? true : false;
  compute_cnt++;
  
  if (skip_exec) return;
  
  // Load bias first : #bias to load = #fmap per dram address 
  //  load 128 bytes (64 x 16-bit for 8-bit fmap / 32 x 32-bit for 16-bit fmap)
  ipa_.adder_buf_b_.Load<DRAM_BANDWIDTH*BIAS_DATA_WIDTH/FMAP_DATA_WIDTH>(
    MemOp(0, 1, 0), bias_ram->BeginPtr(0)
  );
  std::vector<sim_data_type> bias = 
    ipa_.adder_buf_b_.AsVec(0, DRAM_BANDWIDTH/FMAP_DATA_WIDTH, BIAS_DATA_WIDTH);
  // Fetch data from SRAMs to IPA's compute buffers and then compute
  int k = 0;
  int wgt_addr = compute->ker_addr_s;
  int tmp_addr = compute->psum_addr_s;
  // Fetch according to the tiling factors dma_x, dma_y and stride 
  for (int i = compute->dma_y_min; i <= compute->dma_y_max;
           i += compute->read_y_stride) {
    for (int j = compute->dma_x_min; j <= compute->dma_x_max;
           j += compute->read_x_stride) {
      // compute->ker_round = # of output channel / rep_num.
      // e.g: if for every computation round, IPA computes for 8 output channels,
      // ker_round is set to 4 if the output channel block is 32 and in this case
      // the input fm block will be reused 4 times.
      for (int p = 0; p < compute->ker_round; p++) {
        initiate_cnt++;
        // Get fm addr
        int fm_addr = compute->input_addr_s + i * compute->dma_block_x_size + j;
        int pe_bytes = sizeof(IPA_t::PE_buf_t::DType);  // 512 (* 8b)
        int fm_bytes = sizeof(Fm_ram_t::DType);  // 64 (* 8b)
        
        // Load data from sram to ipa rams
        // Replicate fm -> 512 * 8b
        // Replicate input fm to correspond with the PE size(e.g:  if we have 512 bytes per PE address 
        // meaning having 512 macs and 64 bytes per fm address, we replicate the data for fm by 8, in 
        // which case we can compute the data for 8 output channels in parallel. rep_num = 8 in this case.) 
        int rep_num = 8 << compute->copy_mode;
        int valid_bytes = pe_bytes / rep_num;
        fmap_data_type* fm = new fmap_data_type[pe_bytes/(FMAP_DATA_WIDTH/8)];
        //std::cout << "fm cnt after dup : " << pe_bytes/(FMAP_DATA_WIDTH/8) << "\n";
        //std::cout << "rep_num : " << rep_num << "\n";
        //std::cout << valid_bytes << " " << fm_bytes << "\n";
        for (int ii = 0; ii < rep_num; ii++) {
          fmap_data_type* src = reinterpret_cast<fmap_data_type*>(fm_ram->BeginPtr(fm_addr));
          std::memcpy(
            fm + ii * valid_bytes/(FMAP_DATA_WIDTH/8), 
            src + (fm_bytes - valid_bytes)/(FMAP_DATA_WIDTH/8),
            valid_bytes);
        }
        // PE array b/w - 512 PEs with PRECISION bits each
        ipa_.fm_buf_.Load<DRAM_BANDWIDTH*DSP_COUNT/PE_COUNT>(MemOp(0, 1, 0), fm);
        delete [] fm;
    #ifdef DEBUG_DMA_FM
        std::vector<sim_data_type> gdb_ifm = ipa_.fm_buf_.AsVec(0, DSP_COUNT, FMAP_DATA_WIDTH);
        Write2txt<sim_data_type, FMAP_DATA_WIDTH>(DEBUG_DMA_FM_FILENAME, gdb_ifm);
    #endif
        
        // Get wgt addr
        wgt_addr = compute->ker_addr_s + p;
        // Split ker
        fmap_data_type* wgt_src_a = reinterpret_cast<fmap_data_type*>(
            ipa_.wgt_buf_a_.BeginPtr(0));
        fmap_data_type* wgt_src_b = reinterpret_cast<fmap_data_type*>(
            ipa_.wgt_buf_b_.BeginPtr(0));
        fmap_data_type* wgt = reinterpret_cast<fmap_data_type*>(wgt_ram->BeginPtr(wgt_addr));
        for (int ii = 0; ii < pe_bytes; ii++) {
          // For 16-bit, use only a (don't split kernel)
          if(FMAP_DATA_WIDTH == 16) {
            wgt_src_a[ii] = wgt[ii];
          // For 8-bit, split kernel into a and b
          } else if(FMAP_DATA_WIDTH == 8){
            wgt_src_a[ii] = wgt[2 * ii];
            wgt_src_b[ii] = wgt[2 * ii + 1];
          }
        }
        /*std::vector<sim_data_type> gdb = ipa_.wgt_buf_a_.AsVec(0, 512, FMAP_DATA_WIDTH);
        for (int i = 0; i < 16; i++) {
          std::cout << "[" << i << "] ";
          for (int j = 0; j < 32; j++) {
            std::cout << (double)gdb[i*32+j] / (1 << 16)<< " ";
          }
          std::cout << "\n";
        }
        if (p == 1) exit(1);*/
    #ifdef DEBUG_DMA_KER
        std::vector<sim_data_type> gdb_ker = ipa_.wgt_buf_a_.AsVec(0, DSP_COUNT, FMAP_DATA_WIDTH);
        Write2txt<sim_data_type, FMAP_DATA_WIDTH>(DEBUG_DMA_KER_FILENAME, gdb_ker);
    #endif
        // Compute
        ipa_.Forward(2 << compute->output_num);
        //for (auto &p : ipa_.psum) {
        //  std::cout << (double)p / (1 << (11+17)) << " ";
        //}std::cout << "\n";exit(1);

        if (ipa_.GetOutputNum() >= compute->output_channel) {
          // Load out_adder_b
          //   - bias : loaded at the beginning
          //   - temp : partial sum from tmp_buf_
          ipa_.adder_b.clear();
          if (compute->add_bias) {
            for (auto value : bias) {
              ipa_.adder_b.push_back(
                  Saturate(value << compute->shift_num_bias, value > 0, ACCUM_DATA_WIDTH));
            }
          } else if (compute->add_temp) {
            ipa_.adder_buf_b_.Load<DRAM_BANDWIDTH/FMAP_DATA_WIDTH*PSUM_DATA_WIDTH>(MemOp(0, 1, 0),
                tmp_buf_.BeginPtr(tmp_addr));
            std::vector<sim_data_type> tmp = ipa_.adder_buf_b_.AsVec(0, DRAM_BANDWIDTH/FMAP_DATA_WIDTH, PSUM_DATA_WIDTH);
            //if(debug) {
            //  for (auto &x : tmp) std::cout << x << " ";std::cout << "\n";exit(1);
            //}
            for (auto value : tmp) {
              ipa_.adder_b.push_back(static_cast<sim_data_type>(value) << (ACCUM_DATA_WIDTH-PSUM_DATA_WIDTH));  // << 10 for 8-bit
            }
          } else if (compute->add_zero) {  // TODO !!!
            for (int i = 0; i < DRAM_BANDWIDTH/FMAP_DATA_WIDTH; i++) {
              ipa_.adder_b.push_back(0);
            }
          }
          //std::cout << "<<<<" << tmp_addr << ">>>>>\n";
          // Load adder_a from ipa output
          ipa_.Accumulate(
                      compute->shift_num_fm,
                      compute->final_output,
                      tmp_buf_.BeginPtr(tmp_addr), os, dump);
          tmp_addr++;
      #ifdef DEBUG_OUT_ADDER_A
          Write2txt<sim_data_type, 42>(DEBUG_OUT_ADDER_A_FILENAME, ipa_.adder_a_debug);
      #endif
      #ifdef DEBUG_OUT_ADDER_B
          Write2txt<sim_data_type, 42>(DEBUG_OUT_ADDER_B_FILENAME, ipa_.adder_b);
      #endif
      #ifdef DEBUG_PSUM
          Write2txt<sim_data_type, PSUM_DATA_WIDTH>(DEBUG_PSUM_FILENAME, ipa_.res_debug);
      #endif
      #ifdef DEBUG_PSUM_CUT
          if (compute->final_output)
            Write2txt<sim_data_type, FMAP_DATA_WIDTH>(DEBUG_PSUM_CUT_FILENAME, ipa_.cut_debug);
      #endif
        }
      }
    }
  }
  // cycle count
  int64_t ii = 1;
  int64_t epilogue = 30;
  int64_t latency = initiate_cnt * ii + epilogue;
  profiler.collect(latency);
}

void Device::RunComputeFC(OPUComputeInsn* compute) {
  Fm_ram_t* fm_ram = fm_ram_vec_[compute->fm_ram_id];
  Wgt_ram_t* wgt_ram = wgt_ram_vec_[compute->wgt_ram_id];
  Bias_ram_t* bias_ram = bias_ram_vec_[compute->bias_ram_id];
  
  // Load bias first
  ipa_.adder_buf_b_.Load<1024>(MemOp(0, 1, 0), bias_ram->BeginPtr(0));
  std::vector<sim_data_type> bias = ipa_.adder_buf_b_.AsVec(0, 64, 16);
  
  // Control flags
  compute_finish = compute->final_output? true : false;
  compute_cnt++;
  
  int wgt_addr = compute->ker_addr_s;
  int tmp_addr = fc_tmp_addr;
  for (int y = 0; y < 4; y++) {
    ipa_.psum.insert(ipa_.psum.begin(), 0);
  }
  for (int i = compute->dma_y_min; i <= compute->dma_y_max;
         i += compute->read_y_stride) {
    for (int j = compute->dma_x_min; j <= compute->dma_x_max;
         j += compute->read_x_stride) {
        // Get fm addr
        int fm_addr = i * compute->dma_block_x_size + j;
        int pe_bytes = sizeof(IPA_t::PE_buf_t::DType);  // 512 (* 8b)
        int fm_bytes = sizeof(Fm_ram_t::DType);  // 64 (* 8b)
        
        int rep_num = 8 << compute->copy_mode;
        int valid_bytes = pe_bytes / rep_num;
        int8_t* fm = new int8_t[pe_bytes];
        for (int ii = 0; ii < rep_num; ii++) {
          int8_t* src = reinterpret_cast<int8_t*>(fm_ram->BeginPtr(fm_addr));
          std::memcpy(fm+ii*valid_bytes, src + fm_bytes - valid_bytes,
            valid_bytes);
        }
        ipa_.fm_buf_.Load<4096>(MemOp(0, 1, 0), fm);
        delete [] fm;
        /*std::cout << fm_addr << " " << valid_bytes << "\n";
        int8_t* src = reinterpret_cast<int8_t*>(fm_ram->BeginPtr(fm_addr));
        for (int i = 0; i < 64; i ++){
            std::cout << (int)src[i] << " ";
        }
        std::cout << "\n";*/
    #ifdef DEBUG_DMA_FM
        //std::vector<int> gdb = ipa_.fm_buf_.AsVec(0, 512, 8);
        //writeOut<int, 2>(os, gdb, dump);
    #endif
        
        // Get wgt addr
        wgt_addr = compute->ker_addr_s + static_cast<int>(std::ceil(fm_addr/ 4));
        // Split ker
        int8_t* wgt_src_a = reinterpret_cast<int8_t*>(
            ipa_.wgt_buf_a_.BeginPtr(0));
        int8_t* wgt_src_b = reinterpret_cast<int8_t*>(
            ipa_.wgt_buf_b_.BeginPtr(0));
        int8_t* wgt = reinterpret_cast<int8_t*>(wgt_ram->BeginPtr(wgt_addr));
        int offset = (fm_addr % 4) * fm_bytes * 2 * 2;
        for (int ii = 2 * fm_bytes; ii < pe_bytes; ii++) {
          wgt_src_a[ii] = 0;
          wgt_src_b[ii] = 0;
        }
        for (int ii = 0; ii < fm_bytes * 2; ii++) {
          wgt_src_a[ii] = wgt[2 * ii + offset];
          wgt_src_b[ii] = wgt[2 * ii + 1 + offset];
        }
    #ifdef DEBUG_DMA_KER
        //std::vector<int> gdb = wgt_ram->AsVec(wgt_addr, 1024, 8);
        //writeOut<int, 2>(os, gdb, dump);
    #endif
        
        // Compute
        ipa_.Forward(16);
        ipa_.psum.erase(ipa_.psum.begin() + 4, ipa_.psum.begin() + 16);
        // FC accumulation  
        for (int y = 0; y < 4; y++) {
          ipa_.psum[y] += ipa_.psum[y + 4];
        }
        ipa_.psum.erase(ipa_.psum.begin() + 4, ipa_.psum.begin() + 8);
    }
  }
    std::cout << ipa_.GetOutputNum() << " v.s " << compute->output_channel << "\n";
    if (ipa_.GetOutputNum() >= compute->output_channel) {
      // Load out_adder_b
      //   - bias : loaded at the beginning
      //   - temp : partial sum from tmp_buf_
      std::cout << "add_bias: " << compute->add_bias
                << " add_temp: " << compute->add_temp << "\n";
      ipa_.adder_b.clear();
      if (compute->add_bias) {
        for (auto value : bias) {
          ipa_.adder_b.push_back(
              Saturate(value << compute->shift_num_bias, value > 0, 26));
        }
      } else if (compute->add_temp) {
        ipa_.adder_buf_b_.Load<1024>(MemOp(0, 1, 0),
            tmp_buf_.BeginPtr(tmp_addr));
        std::vector<sim_data_type> tmp = ipa_.adder_buf_b_.AsVec(0, 64, 16);
        for (auto value : tmp) {
          ipa_.adder_b.push_back(static_cast<sim_data_type>(value) << 10);
        }
      }

      // Load adder_a from ipa output
      ipa_.Accumulate(
                  compute->shift_num_fm,
                  compute->final_output,
                  tmp_buf_.BeginPtr(tmp_addr), os, dump);
      std::cout << "tmp buffer addr: " << fc_tmp_addr << "\n";
      fc_tmp_addr = ++tmp_addr;
      if (fc_tmp_addr == compute->output_block_y_size * compute->output_block_x_size) {
        fc_tmp_addr = 0;
      }
    }
  profiler.collect(1);
}  


void Device::RunComputeDW(OPUComputeInsn* compute) {
  Fm_ram_t* fm_ram = fm_ram_vec_[compute->fm_ram_id];
  Wgt_ram_t* wgt_ram = wgt_ram_vec_[compute->wgt_ram_id];
  Bias_ram_t* bias_ram = bias_ram_vec_[compute->bias_ram_id];
  // wgt_ram --> wgt_ram_dw_
  // 1024 * 8--> 2048 * 8 (64*32 filled from 64*18)
  int idx = 0;
  int tmp = 0;
  int8_t* wgt_dw = new int8_t[2048];
  for (int i = 0; i < 64; i++) {
    int8_t* wgt = reinterpret_cast<int8_t*>(wgt_ram->BeginPtr(i));
    for (int j = 0; j < 16; j++) {
      std::memcpy(wgt_dw + idx * 64, wgt + j * 64, 64);
      if (idx == 17) {
        std::memset(wgt_dw + 18 * 64, 0, 14*64);
        MemOp mem = MemOp(
            0,
            1,
            tmp++);
        wgt_ram_dw_.Load<2048>(mem, wgt_dw);
      }
      idx = (idx + 1) % 18;
    }
  }
  delete [] wgt_dw;
  // Instrumentation
  int64_t initiate_cnt = 0;
  // Control flags
  compute_finish = compute->final_output? true : false;
  compute_cnt++;
  
  if (skip_exec) return;
  
  // Load bias first
  ipa_.adder_buf_b_.Load<1024>(MemOp(0, 1, 0), bias_ram->BeginPtr(0));
  std::vector<sim_data_type> bias = ipa_.adder_buf_b_.AsVec(0, 64, 16);
  // Fetch data from SRAMs to IPA's compute buffers and then compute
  int k = 0;
  int wgt_addr = compute->ker_addr_s;
  int tmp_addr = 0;
  // To be more accurate, we should follow dma_min/max and write them to 64 parallel buffer
  // first. Then we fetch 3x3 data window from line buffer.
  // Here I skip the middle part, so use (y_max-ky+1) as dma access upper bound
  // not accurate but works for now for stride=1
  for (int i = compute->dma_y_min; i <= compute->dma_y_max - compute->ker_y_size + 1;
           i += compute->read_y_stride) {
    for (int j = compute->dma_x_min; j <= compute->dma_x_max - compute->ker_x_size + 1;
           j += compute->read_x_stride) {
      for (int p = 0; p < compute->ker_round; p++) {
        initiate_cnt++;
        continue;
        // Get fm addr
        int fm_addr_st = i * compute->dma_block_x_size + j;
        int pe_bytes = sizeof(IPA_t::PE_buf_t::DType);  // 16*64 (* 8b)
        int fm_bytes = sizeof(Fm_ram_t::DType);  // 64 (* 8b)
        
        // Load data from sram to ipa rams via line buffer
        int8_t* fm = new int8_t[pe_bytes];
        std::memset(fm, 0, pe_bytes);      
        int ker_cnt = compute->ker_y_size * compute->ker_x_size;
        for (int kh = 0; kh < compute->ker_y_size; kh++) {
          for (int kw = 0; kw < compute->ker_x_size; kw++) {
            int fm_addr = fm_addr_st + kh * compute->dma_block_x_size + kw;
            // split to fill 16 inputs of PE, 64 in total
            // for example, {9 elements, (7){0}} x 64 for ipa_.fm_buf_
            int8_t* src = reinterpret_cast<int8_t*>(fm_ram->BeginPtr(fm_addr));
            for (int ii = 0; ii < 64; ii++) {
              int byte_idx = ii * 16 + kh * compute->ker_x_size + kw;
              fm[byte_idx] = src[ii];
            }
          }
        }
        ipa_dw_.fm_buf_.Load<8192>(MemOp(0, 1, 0), fm);
        // select 9 from 16 for each PE (16*64 -> 9*64) for debug output
        /*
        std::vector<int> gdb;
        std::vector<int> tmp = ipa_.fm_buf_.AsVec(0, 1024, 8);
        for (int ii = 0; ii < 64; ii++) {
          for (int jj = 0; jj < 9; jj++) {
            gdb.push_back(tmp[ii * 16 + jj]);
          }
        }
        writeOut<int, 2>(os, gdb, dump);
        */
        
        // Get wgt addr
        wgt_addr = compute->ker_addr_s + p;
        // Split ker
        int8_t* wgt_src_a = reinterpret_cast<int8_t*>(
            ipa_dw_.wgt_buf_a_.BeginPtr(0));
        int8_t* wgt_src_b = reinterpret_cast<int8_t*>(
            ipa_dw_.wgt_buf_b_.BeginPtr(0));
        int8_t* wgt = reinterpret_cast<int8_t*>(wgt_ram_dw_.BeginPtr(wgt_addr));
        int jj = 0;
        for (int ii = 0; ii < 64 * ker_cnt; ii++) {
          wgt_src_a[jj] = wgt[2 * ii];
          wgt_src_b[jj] = wgt[2 * ii + 1];
          if ((jj + 1) % ker_cnt == 0) {
            for (int u = 0; u < 16 - ker_cnt; u++) {
              wgt_src_a[jj] = 0;
              wgt_src_b[jj] = 0;
              jj++;
            }  
          } else {
            jj++;
          }
        }
        /*
        std::vector<int> tmp = wgt_ram->AsVec(wgt_addr, 2048, 8);
        std::vector<int> gdb(tmp.begin(), tmp.begin() + 1152);
        writeOut<int, 2>(os, gdb, dump);
        */
        
        // Compute
        ipa_dw_.Forward(2 << compute->output_num);

        if (ipa_.GetOutputNum() >= compute->output_channel) {
          // Load out_adder_b
          //   - bias : loaded at the beginning
          //   - temp : partial sum from tmp_buf_
          ipa_dw_.adder_b.clear();
          if (compute->add_bias) {
            for (auto value : bias) {
              ipa_dw_.adder_b.push_back(
                  Saturate(value << compute->shift_num_bias, value > 0, 26));
            }
          } else if (compute->add_temp) {
            ipa_dw_.adder_buf_b_.Load<1024>(MemOp(0, 1, 0),
                tmp_buf_.BeginPtr(tmp_addr));
            std::vector<sim_data_type> tmp = ipa_dw_.adder_buf_b_.AsVec(0, 64, 16);
            for (auto value : tmp) {
              ipa_dw_.adder_b.push_back(static_cast<sim_data_type>(value) << 10);
            }
          }

          // Load adder_a from ipa output
          ipa_dw_.Accumulate(
                      compute->shift_num_fm,
                      compute->final_output,
                      tmp_buf_.BeginPtr(tmp_addr), os, dump);
          tmp_addr++;
        }
      }
    }
  }
  
  // cycle count
  int64_t ii = 1;
  int64_t epilogue = 30;
  int64_t latency = compute->dma_block_x_size*2 +2 + initiate_cnt * ii + epilogue;
  profiler.collect(latency);
}

/*
 * Top function wrapper
 */
void Device::Run(int parallel_block_cnt) {
  for (int u = 0; u < parallel_block_cnt; u++) {
    LOG_INFO << "sync id = " << u + 1;
    //std::cout <<
    //    "*************************Parallel Block**********************\n";
    if (!event_q.empty()) {
      std::vector<OPUGenericInsn*> ins_vec = event_q.front();
      event_q.pop();
      // Arbiter
      for (auto ins : ins_vec) {
        if (ins->IsInstance<OPUComputeInsn>()) {
          OPUComputeInsn* compute = static_cast<OPUComputeInsn*>(ins);
          int fid = (fm_ram_id + fm_ram_vec_.size() - 1)
            % fm_ram_vec_.size();
          int wid = (wgt_ram_id + wgt_ram_vec_.size() - 1)
            % wgt_ram_vec_.size();
          int bid = (bias_ram_id + bias_ram_vec_.size() - 1)
            % bias_ram_vec_.size();
          // Consecutive dma for nxn conv, fetching from same fm buffer
          if (reg_.dma_start_trig == OPU_DMA_TRIG_DMA) {
            fid = compute->fm_ram_id;
            wid = compute->wgt_ram_id;
            bid = compute->bias_ram_id;
          }
          compute->SwitchUpdate(fid, wid, bid);
        }
      }
      // Run modules in parallel semantic (seq in real)
      OPUComputeInsn* cinst = nullptr;
      for (auto ins : ins_vec) {
        if (ins->IsInstance<OPUDDRLDInsn>()) {
          OPUDDRLDInsn* load = static_cast<OPUDDRLDInsn*>(ins);
          RunLoad(load);
        } else if (ins->IsInstance<OPUDDRSTInsn>()) {
          OPUDDRSTInsn* store = static_cast<OPUDDRSTInsn*>(ins);
          RunStore(store);
        } else if (ins->IsInstance<OPUComputeInsn>()) {
          OPUComputeInsn* compute = static_cast<OPUComputeInsn*>(ins);
          RunCompute(compute);
          cinst = compute;
        } else {
          std::cout << "[ERROR] unknown instruction type!";
          exit(1);
        }
      }
      FetchInsn();
      // Control flags - simulate the end of compute event
      if (cinst != nullptr) {
        //compute_finish = cinst->final_output? true : false;
      }
      profiler.sync();
      // Propagate completion
      std::vector<OPUGenericInsn*> ins_vec_next = DependencyCheck(ins_vec);
      if (ins_vec_next.size() > 0) {
        event_q.push(ins_vec_next);
      }
    } else {
      compute_finish = false;
      compute_cnt = 0;
      compute_cnt_enable = false;
      break;
    }
  }
  profiler.dump();
}

/*
 * Propagate completion and get new events to run
 */
std::vector<OPUGenericInsn*>
    Device::DependencyCheck(std::vector<OPUGenericInsn*> ins_vec) {
  std::stringstream os;
  std::vector<OPUGenericInsn*> ins_vec_next;
  for (auto ins : ins_vec) {
    os << GetInsName(ins) << "\n";
    // Propagate completion of ins to its successors
    for (auto succ : ins->succ_) {
      os << "[check succ]" << GetInsName(succ) << "\n";
      succ->pred_[ins] = true;
      bool sat = false;
      if (succ->pred_.size() > 0) {
        sat = true;
        for (auto item : succ->pred_) {
          sat &= item.second;
          os << "\t[succ pred]" << GetInsName(item.first) << ":"
             << item.second << "\n";
        }
      }
      // Check pre-condition assertions
      if (sat) {
        os << "\tcheck " << succ->assert_.size() << " assertion(s)\n";
        if (succ->assert_.size() > 0) {
          os << "compute_cnt = " << compute_cnt << "\n";            
          for (auto item : succ->assert_) {
            if (static_cast<bool*>(item.first)) {
              if (static_cast<int>(*static_cast<bool*>(item.first))
                  != item.second) {
                os << "\t$" << static_cast<int>(*static_cast<bool*>(item.first))
                   << " != " << item.second << "\n";
                sat = false;
                break;
              }
            } else if (static_cast<int*>(item.first)) {
              if (*static_cast<int*>(item.first) != item.second) {
                os << "\t" << *static_cast<int*>(item.first) << " != "
                   << item.second << ">>" << compute_cnt << "\n";
                sat = false;
                break;
              } else {
                os << "\treset int\n";
                compute_cnt = 0;
                compute_cnt_enable = false;
              }
            } else {
              os << "\tELSE\n";
            }
          }
        }
      }
      // If all pre-conditions satisfied, schedule succ to next
      if (sat) {
        auto it = std::find(ins_vec_next.begin(),
                            ins_vec_next.end(),
                            succ);
        if (it == ins_vec_next.end()) {
          ins_vec_next.push_back(succ);
        }
      }
    }
  }
  if (compute_cnt == reg_.ddr_load_start_dma_num) {
    compute_cnt = 0;
  }
  // if (dump) std::cout << os.str();
  return ins_vec_next;
}

/*
 * Fetch one instruction block
 */
void Device::FetchInsn() {
  //std::cout << "************BEGIN OF ONE INS BLK************\n";
  int ins_rd_cnt = 0;
  // Fetch until ins->immi == 0
  while (1) {
    //std::cout << "ins_ram addr: " << ins_pc << "\n";
    ins_rd_cnt++;
    // Fetch one short instruction
    uint32_t data;
  #ifdef SIMULATE_INS_RAM
    Ins_ram_t *ins_ram = ins_ram_vec_[ins_ram_id_r];
    data = *static_cast<uint32_t*>(ins_ram->BeginPtr(ins_pc_local));
    LOG_INFO << "fetch ins from inst_buffer[" << ins_ram_id_r << "][" << ins_pc_local << "]"
      << " : " << std::hex << data << "\n";    
    ins_pc_local++;
    if (ins_pc_local == INS_BUFFER_DEPTH) {
      ins_pc_local = 0;
      ins_ram_id_r = (ins_ram_id_r + 1) % ins_ram_vec_.size();
      LOG_INFO << "next read instruction from inst_buffer[" << ins_ram_id_r << "]\n";
    }
  #else
    data = *static_cast<uint32_t*>(ins_ram_.BeginPtr(ins_pc));
  #endif
    OPUShortInsn* ins = new OPUShortInsn(data);
    // Decode and update register file
    ins->Decode(&reg_);
    // Update control flow
    DependencyUpdate(ins);
    ins_pc++;
    if (!ins->immi) {
      load_ins_->ReadFrom(&reg_);
      store_ins_->ReadFrom(&reg_);
      compute_ins_->ReadFrom(&reg_);
      DependencyUpdateUtil();
      if (layer_start_) {
        event_q.push({load_ins_});
        layer_start_ = false;
      }
      break;
    }
  }
  LOG_INFO << "trig load(" << reg_.ddr_load_start_trig << ") "
    << "compute(" << reg_.dma_start_trig << ") "
    << "store(" << reg_.ddr_save_start_trig << ")\n";
  //std::cout << "************END OF ONE INS BLK************\n";
  
  // cycle count
  profiler.collect(ins_rd_cnt);
}

/*
 * Check if the whole model is completed
 */
bool Device::IsComplete() {
  return reg_.network_done;
}

/*
 * Parse trigger conditions embedded in instructions
 */
void Device::DependencyUpdate(OPUShortInsn* ins) {
  if (ins->opcode == 11) {
    load_ins_->SetAssert({});
    if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_DDRLD) {
      load_ins_->SetPred({load_ins_});
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_DDRLD_DMA) {
      load_ins_->SetPred({load_ins_, compute_ins_});
      load_ins_->SetAssert(
        {{static_cast<void*>(&compute_cnt), reg_.ddr_load_start_dma_num }});
      compute_cnt_enable = true;
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_LAYER_START) {
      layer_start_ = true;
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_DDRST) {
      load_ins_->SetPred({store_ins_});
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_BRAMST) {
      load_ins_->SetPred({compute_ins_});
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_BRAMWB_DDRLD) {
      load_ins_->SetPred({load_ins_, store_ins_});
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_DDRLD_DDRST) {
      load_ins_->SetPred({load_ins_, store_ins_});
    } else {
      load_ins_->SetPred({});
    }
  } else if (ins->opcode == 19) {
    store_ins_->SetAssert({});
    if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_BRAMST_DDRLD) {
      store_ins_->SetPred({compute_ins_/*, load_ins_*/});
      store_ins_->SetAssert({{static_cast<void*>(&compute_finish), 1}});
    } else if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_DDRST) {
      store_ins_->SetPred({store_ins_});
    } else if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_BRAMWB_DDRLD) {
      store_ins_->SetPred({load_ins_, store_ins_});
    } else if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_DDRLD) {
      store_ins_->SetPred({load_ins_});
    } else if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_DDRLD_DDRST) {
      store_ins_->SetPred({load_ins_, store_ins_});
    } else if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_SINGLE_DDRLD) {
      store_ins_->SetPred({load_ins_});
      store_ins_->SetAssert(
        {{static_cast<void*>(&load_single), 1}});
    } else {
      store_ins_->SetPred({});
    }
  } else if (ins->opcode == 12) {
    compute_ins_->SetAssert({});
    if (reg_.dma_start_trig == OPU_DMA_TRIG_DMA) {
      compute_ins_->SetPred({compute_ins_});
    } else if (reg_.dma_start_trig == OPU_DMA_TRIG_DDRLD) {
      compute_ins_->SetPred({load_ins_});
    } else if (reg_.dma_start_trig == OPU_DMA_TRIG_DMA_DDRLD) {
      compute_ins_->SetPred({compute_ins_});
      compute_ins_->SetAssert(
        {{static_cast<void*>(&compute_cnt), reg_.ddr_load_start_dma_num }});
      compute_cnt_enable = true;
    } else if (reg_.dma_start_trig == OPU_DMA_TRIG_DDRST_NOT_1_6) {
      compute_ins_->SetPred({store_ins_});
    } else if (reg_.dma_start_trig == OPU_DMA_TRIG_DDRST) {
      compute_ins_->SetPred({store_ins_});
    } else {
      compute_ins_->SetPred({});
    }
  }
}

/*
 * Update events to trigger
 */
void Device::DependencyUpdateUtil() {
  for (auto ins : ins_vec_) {
    ins->succ_.clear();
  }
  for (auto ins : ins_vec_) {
    for (auto item : ins->pred_) {
      item.first->succ_.push_back(ins);
    }
  }
}

/*
 * Get instruction name
 */
std::string Device::GetInsName(OPUGenericInsn* ins) {
  if (ins->IsInstance<OPUDDRLDInsn>()) {
    return "DDR_LD_INSN";
  } else if (ins->IsInstance<OPUDDRSTInsn>()) {
    return "DDR_ST_INSN";
  } else if (ins->IsInstance<OPUComputeInsn>()) {
    return "COMPUTE";
  } else {
    return "UNKNOWN";
  }
}
