#include <string>
#include <fstream>
#include <queue>
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <climits>
#include <cassert>

#include "./hw_spec.h"
#include "./instruction.h"
#include "./bit_util.h"
#include "./vmem.h"
#include "./config.h"
#include "./profiler.h"
#include "./nonlinear.h"
#include "./smem.h"
#include "./logging.h"
#include "./debug_config.h"

#ifndef FSIM_ACCELERATOR_H_
#define FSIM_ACCELERATOR_H_
/*
 * Define template components for hardware modules, including SRAM, IPA
 */

/*
 * IPA class is parameterized with
 *  - mac : number of multiply-accumulate units (= DSP macros on FPGA) per PE
 *  - pe : number of PE (processing unit), which is a bunch of macs followed 
 *         by adder tree for reduction
 *  - operand_bits : word lengths for mac operands (8 by default)
 */
template<int mac_per_pe, int pe, int operand_bits>
class IPA {
 public:
  static const int nMac = mac_per_pe * pe; // mac data width unspecified, to be doubled if decomposed
  // Compute buffer
  using PE_buf_t = SRAM<mac_per_pe*operand_bits, 1, pe>;
  PE_buf_t fm_buf_;
  PE_buf_t wgt_buf_a_;
  PE_buf_t wgt_buf_b_;
  using Acc_buf_t = SRAM<32*64, 1, 1>;
  Acc_buf_t adder_buf_a_;
  Acc_buf_t adder_buf_b_;

  std::vector<sim_data_type> psum;
  std::vector<sim_data_type> adder_b;
  std::vector<sim_data_type> adder_a_debug;
  std::vector<sim_data_type> res_debug;
  std::vector<sim_data_type> cut_debug;

  bool debug = false;

  // Get current valid ipa output number
  int GetOutputNum() {
    return psum.size();
  }

  // Fetch data from compute buffer and get ipa results
  void Forward(int output_num) {
    std::vector<sim_data_type> ifm = fm_buf_.AsVec(0, nMac, FMAP_DATA_WIDTH);
    std::vector<sim_data_type> wgt_a = wgt_buf_a_.AsVec(0, nMac, FMAP_DATA_WIDTH);
    std::vector<sim_data_type> wgt_b = wgt_buf_b_.AsVec(0, nMac, FMAP_DATA_WIDTH);
    /*if (debug) {
      for (int i = 0; i < 32; i++) std::cout << (double)ifm[i] / (1<<15) << " "; std::cout <<"\n\n";
      for (int i = 0; i < 16; i++) {
        std::cout << "[" << i << "] ";
        for (int j = 0; j < 32; j++) {
          std::cout << (double)wgt_a[i * 32 + j] / (1<<16)<< " ";
        }std::cout << "\n";
      }
    }*/
    std::vector<sim_data_type> psum_local;
    int num = (FMAP_DATA_WIDTH == 8) ? output_num / 2 : output_num;
    size_t step = nMac / num;
    for (int i = 0; i < num; i++) {
      sim_data_type sum = 0;
      for (int ii = 0; ii < step; ii++) {
        sum += ifm[i * step + ii] * wgt_a[i * step + ii];
      }
      //sum = std::inner_product(ifm.begin() + i * step,
      //        ifm.begin() + (i + 1) * step, wgt_a.begin() + i * step, 0);
      psum_local.push_back(sum);
      if (FMAP_DATA_WIDTH == 16)
        continue;
      sum = 0;
      for (int ii = 0; ii < step; ii++) {
        sum += ifm[i * step + ii] * wgt_b[i * step + ii];
      }
      //sum = std::inner_product(ifm.begin() + i * step,
      //        ifm.begin() + (i + 1) * step, wgt_b.begin() + i * step, 0);
      psum_local.push_back(sum);
    }

    // Concatenate at front
    //psum.insert(psum.begin(), psum_local.begin(), psum_local.end());
    psum.insert(psum.end(), psum_local.begin(), psum_local.end());
  }

  // Output control for accumulation
  void Accumulate(int fm_lshift_num, bool cut, void* dst,
    std::ofstream& os, bool dump) 
  {
    std::vector<sim_data_type> adder_a(2*pe/(FMAP_DATA_WIDTH/8) - psum.size(), 0);
    for (int i = 0; i < psum.size(); i++) {
      sim_data_type value = psum[i];
      adder_a.push_back(
        Saturate(value << fm_lshift_num, value > 0, ACCUM_DATA_WIDTH));
    }
    adder_a_debug = adder_a;
    // std::vector<int> gdb = wgt_ram.AsVec(wgt_addr, 1024, 8);
    psum.clear();
    std::vector<sim_data_type> res;
    auto ia = adder_a.begin();
    auto ib = adder_b.begin();
    for (int ii = 0; ii < 2*pe/(FMAP_DATA_WIDTH/8); ii++) {
      sim_data_type p = *(ia++) + *(ib++);
      res.push_back(p);
    }
    // 26b--cut->16b
    for (auto &item : res) {
      item = Saturate(item >> (ACCUM_DATA_WIDTH - PSUM_DATA_WIDTH), item > 0, PSUM_DATA_WIDTH);  // >> 10
    }
    res_debug = res;
    if (cut) {
      // 16b--round->8b
      for (auto &item : res) {
        bool positive = item > 0;
        if (ROUND_MASK(item))
          item = (item >> FMAP_DATA_WIDTH) + 1;
        else
          item = item >> FMAP_DATA_WIDTH;
        item = Saturate(item, positive, FMAP_DATA_WIDTH);
      }
      cut_debug = res;
    }
    // write to dst* of temp buffer
    std::vector<psum_data_type> trunc;
    int j = 0;
    for (auto item : res) {
      // -> little endian
      psum_data_type value = static_cast<psum_data_type>(item);
      // psum 32bit
      j++;
      // std::cout << (double)value / ((int64_t)1<<(15+16+fm_lshift_num-(ACCUM_DATA_WIDTH-PSUM_DATA_WIDTH))) << " ";
      // cut -> 16bit
      //if (cut) std::cout << (double)value / ((int64_t)1<<(15+16+fm_lshift_num-(ACCUM_DATA_WIDTH-PSUM_DATA_WIDTH)-FMAP_DATA_WIDTH)) << " ";
      trunc.push_back(REVERSE_BYTES(value));
    }//if(debug){std::cout << "\n";}
    std::memcpy(dst, &trunc[0], DRAM_BANDWIDTH/FMAP_DATA_WIDTH*PSUM_DATA_WIDTH/8);  // byte count
  }
};

/*
 * Device class is the top abstraction of OPU overlay
 */
class Device {
 public:
  uint32_t ins_pc{0};
#ifdef SIMULATE_INS_RAM
  uint32_t ins_pc_local{0};
#endif
  // Special purpose registers
  OPURegFile reg_;
  // Scratchpad memory
  using Ins_ram_t = SRAM<INS_BUFFER_WIDTH, INS_BUFFER_DEPTH, 1>;
  using Fm_ram_t = SRAM<DRAM_BANDWIDTH, FMAP_BUFFER_DEPTH, 1>;
  // weight buffer bandwidth (= DRAM_BANDWIDTH x DSP_COUNT/PE_COUNT) matches the input bandwidth of mac array
  using Wgt_ram_t = SRAM<DRAM_BANDWIDTH, WGT_BUFFER_DEPTH, DSP_COUNT/PE_COUNT>;
  using Bias_ram_t = SRAM<DRAM_BANDWIDTH/FMAP_DATA_WIDTH*BIAS_DATA_WIDTH, 1, 1>;
  Ins_ram_t ins_ram_;
  Ins_ram_t ins_ram_b_;
  Fm_ram_t fm_ram_a_;
  Fm_ram_t fm_ram_b_;
  Wgt_ram_t  wgt_ram_a_;
  Wgt_ram_t  wgt_ram_b_;
  Bias_ram_t bias_ram_a_;
  Bias_ram_t bias_ram_b_;
  // IPA
  using IPA_t = IPA<DSP_COUNT/PE_COUNT, PE_COUNT, FMAP_DATA_WIDTH>;
  IPA_t ipa_;
  // Partial sum buffer
  using Psum_t = SRAM<DRAM_BANDWIDTH/FMAP_DATA_WIDTH*PSUM_DATA_WIDTH, PSUM_BUFFER_DEPTH, 1>;
  Psum_t tmp_buf_;
  // DDR
  using DRAM = VirtualMemory;
  DRAM dram;
  // Nonlinear units
  ActivationUnit<DRAM_BANDWIDTH/FMAP_DATA_WIDTH> activation_unit_;
  PoolingUnit<DRAM_BANDWIDTH/FMAP_DATA_WIDTH> pooling_unit_;
  ElementwiseALU<DRAM_BANDWIDTH/FMAP_DATA_WIDTH> elw_unit_;
  using NVM_t = NVM<8, DRAM_BANDWIDTH, 32>;
  NVM_t nvm_; 
  
  // dw
  using Wgt_ram_dw_t = SRAM<1024, 64, 16>;
  Wgt_ram_dw_t wgt_ram_dw_;
  using IPA_DW_t = IPA<16, 64, 8>;
  IPA_DW_t ipa_dw_;

  // Double buffering
  std::vector<Fm_ram_t*> fm_ram_vec_;
  std::vector<Wgt_ram_t*> wgt_ram_vec_;
  std::vector<Bias_ram_t*> bias_ram_vec_;
  std::vector<Ins_ram_t*> ins_ram_vec_;
  size_t fm_ram_id{0};
  size_t wgt_ram_id{0};
  size_t bias_ram_id{0};
  size_t ins_ram_id{0};
  size_t ins_ram_id_r{0};
  size_t param_ram_id{0};

  // Control flow
  std::queue<std::vector<OPUGenericInsn*>> event_q;
  OPUDDRLDInsn* load_ins_;
  OPUDDRSTInsn* store_ins_;
  OPUComputeInsn* compute_ins_;
  std::vector<OPUGenericInsn*> ins_vec_;
  void DependencyUpdate(OPUShortInsn* ins);
  void DependencyUpdateUtil();
  std::vector<OPUGenericInsn*>
    DependencyCheck(std::vector<OPUGenericInsn*> ins_vec);
  // Global variables
  bool layer_start_{false};
  bool compute_finish {false};
  bool compute_cnt_enable {false};
  int compute_cnt {0};
  bool load_single {false};
  int fc_tmp_addr {0};

  // Function wrappers
  void Run(int cnt = INT_MAX);
  void FetchInsn();
  void LoadIns(int ddr_ins_addr);  // load instruction from ddr to on-chip inst ram
  void RunInsn(OPUGenericInsn* ins);
  void RunLoad(OPUDDRLDInsn* load);
  void RunStore(OPUDDRSTInsn* store);
  void RunPostProcess(OPUDDRSTInsn* store);
  void RunPostProcess_legacy(OPUDDRSTInsn* store);
  void RunPostProcess_nvm(OPUDDRSTInsn* store);
  void RunPostProcess_transpose(OPUDDRSTInsn* store);
  void RunPadding(OPUDDRSTInsn* store);
  void RunCompute(OPUComputeInsn* compute);
  void RunComputeDW(OPUComputeInsn* compute);
  void RunComputeFC(OPUComputeInsn* compute);
  // Sub-function wrappers
  void Pooling(std::vector<sim_data_type>& data_o, std::vector<sim_data_type> data_i, int type,
    bool st, bool ed, int window_size);
  void Activation(std::vector<sim_data_type>& data, int type);
  void ResidualAdd(std::vector<sim_data_type>& data, bool enable, int addr);

  // Utility function
  std::string GetInsName(OPUGenericInsn* ins);
  bool IsComplete();

  // Debug
  std::ofstream os;
  bool dump {false};
  bool skip_exec {false};
  Profiler profiler;
  bool debug {false};

  std::unordered_map<std::string, std::ofstream> ofile_name_map_;
  template<typename... Args> 
  void OpenOfile(Args... filenames) {
    std::vector<std::string> fv = {filenames...};
    for (auto &name : fv) {
      ofile_name_map_[name] = std::ofstream(name);
    }
  }
  void CloseAllOfile() {
    for (auto &item : ofile_name_map_) {
      item.second.close();
    }
  }
  template <typename T, int W>
  void Write2txt(std::string key, std::vector<T> data) {
    auto it = ofile_name_map_.find(key);
    assert(it != ofile_name_map_.end());
    bool debug = false;//key == DEBUG_OUT_ADDER_B_FILENAME;
    writeOut<T, W>(it->second, data, true, debug);
  }

  Device() {
    //dram.Init();
    // srams
    //ins_ram_.InitFromFile(INS_FILE_PATH);
    ins_ram_vec_ = {&ins_ram_, &ins_ram_b_};
    fm_ram_vec_ = {&fm_ram_a_, &fm_ram_b_};
    wgt_ram_vec_ = {&wgt_ram_a_, &wgt_ram_b_};
    bias_ram_vec_ = {&bias_ram_a_, &bias_ram_b_};
    // coarse-grained generic instructions
    load_ins_ = new OPUDDRLDInsn();
    store_ins_ = new OPUDDRSTInsn();
    compute_ins_ = new OPUComputeInsn();
    ins_vec_ = {load_ins_, store_ins_, compute_ins_};
    
    os.open("out.txt");
  }
};
#endif  // FSIM_ACCELERATOR_H_
