#ifndef FSIM_SMEM_H_
#define FSIM_SMEM_H_

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

#include <glog/logging.h>

#include "./hw_spec.h"
#include "./bit_util.h"
#include "./config.h"

/*
 * MemOp class encapsulates source/destination address and size for memory operation
 */ 
class MemOp {
 public:
  uint32_t src_addr;
  uint32_t size;
  uint32_t dst_addr;
  MemOp(uint32_t src_addr, uint32_t size, uint32_t dst_addr) {
    this->src_addr = src_addr;
    this->size = size;
    this->dst_addr = dst_addr;
  }
};

/*
 * reference: tsim for vta
 *   https://github.com/apache/incubator-tvm-vta/blob/master/src/sim/sim_driver.cc
 * 
 * SRAM class is parameterized with 
 *   - bits : bits per address for one bank
 *   - size : total address number
 *   - banks : memory banks number
 *   (assume data at same address from different banks are concatenated)
 */
template<int bits, int size, int banks>
class SRAM {
 public:
  static const int dBytes = bits * banks / 8;
  using DType = typename std::aligned_storage<dBytes, dBytes>::type;

  SRAM() {
    data_ = new DType[size];
  }

  ~SRAM() {
    delete [] data_;
  }

  // Get the i-th index
  void* BeginPtr(uint32_t index) {
    // CHECK_LT(index, kMaxNumElem);
    return &(data_[index]);
  }

  // Copy data from disk file to SRAM
  void InitFromFile(std::string filename) {
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    CHECK_LE(fileSize, size * dBytes);
    file.seekg(0, std::ios::beg);
    // read the data:
    file.read(reinterpret_cast<char*>(data_), fileSize);
    file.close();
  }

  // Copy data from disk file to SRAM with address offset and size specified
  //   - src_wl : word length corresponding to mem.src and mem.size
  template<uint32_t src_wl>
  void LoadFromFile(MemOp mem, const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    file.seekg(mem.src_addr * src_wl / 8, std::ios::beg);
    int bytes = mem.size * src_wl / 8;
    file.read(reinterpret_cast<char*>(data_) +
        mem.dst_addr * dBytes, bytes);
    file.close();
  }

  // Copy data from in-memory pointer to SRAM
  template<int src_bits>
  void Load(MemOp mem, void* src) {
    int8_t* src_t = reinterpret_cast<int8_t*>(src);
    size_t bytes = mem.size * src_bits / 8;
    std::memcpy(data_ + mem.dst_addr, src_t, bytes);
  }

  void SetZero(int addr_s, int addr_e) {
    DType* src = data_ + addr_s;
    int8_t* t = reinterpret_cast<int8_t*>(src);
    std::vector<int8_t> zero(dBytes, 0);
    for (int i = 0; i < addr_e - addr_s; i++) {
      std::memcpy(&t[i*dBytes], &zero[0], dBytes);
    }
  }

  // Get data as std::vector<int> to compute
  std::vector<sim_data_type> AsVec(int addr, int vec_size, int src_wl) {
    sim_data_type* data = new sim_data_type[vec_size];
    DType* src = data_ + addr;
    for (int i = 0; i < vec_size; i++) {
      if (src_wl == 8) {
        data[i] = static_cast<sim_data_type>((reinterpret_cast<int8_t*>(src))[i]);
      } else if (src_wl == 16) {
        int16_t value = (reinterpret_cast<int16_t*>(src))[i];
        value = ((value & 0xFF) << 8) | ((value & 0xFF00) >> 8);
        data[i] = static_cast<sim_data_type>(value);
      } else if (src_wl == 32) {
        int32_t value = (reinterpret_cast<int32_t*>(src))[i];
        value = REVERSE_BYTES32(value);
        data[i] =static_cast<sim_data_type>(value);
      }
    }
    std::vector<sim_data_type> vec(vec_size);
    std::memcpy(&vec[0], data, sizeof(sim_data_type)*vec_size);
    delete [] data;
    return vec;
  }

 private:
  DType* data_;
};

#endif
