#include "./vmem.h"
#include "cnpy.h"
#include "hw_spec.h"
#include <sys/stat.h>
#include <glog/logging.h>

void VirtualMemory::FromFile(void* dst, size_t bytes, std::string filename, std::string info) {
  // open the file:
  std::ifstream file(filename.c_str(), std::ios::binary);
  // get its size:
  file.seekg(0, std::ios::end);
  size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  CHECK(fileSize == bytes) << info << " Get:" << fileSize << " v.s. Target:" << bytes;
  // read the data:
  file.read(reinterpret_cast<char*>(dst), fileSize);
  file.close();
}
 
void VirtualMemory::FromFile(addr_t phy_addr, std::string filename) {
  void* dst = GetAddr(phy_addr);
  // open the file:
  std::ifstream file(filename.c_str(), std::ios::binary);
  // get its size:
  file.seekg(0, std::ios::end);
  size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  // read the data:
  file.read(reinterpret_cast<char*>(dst), fileSize);
  file.close();
}

void VirtualMemory::Init() {
  base = Alloc((ins_addr_ini - fm_addr_ini) *64);
  fm = GetAddr(fm_addr_ini);
  wgt = GetAddr(wgt_addr_ini);
  bias = GetAddr(bias_addr_ini);
  /*
  FromFile(fm, 416*416*64, "tinyyolo/ifm.bin");
  //FromFile(wgt, 246336*64, "tinyyolo/weights.bin");
  //FromFile(bias, 51648, "tinyyolo/bias.bin");
  FromFile(wgt, 61897728, "yolov3/weights.bin");
  FromFile(bias, 301640, "yolov3/bias.bin");
  */
  FromFile(fm, IFM_BYTES, IFM_FILE_PATH, "FM");
  FromFile(wgt, WGT_BYTES, WGT_FILE_PATH, "WEIGHT");
  FromFile(bias, BIAS_BYTES, BIAS_FILE_PATH, "BIAS");
}

void* VirtualMemory::Alloc(addr_t size) {
  addr_t npages = (size + kPageSize - 1) / kPageSize;
  addr_t start = ptable_.size();
  std::unique_ptr<Page> p(new Page(start, npages));
  ptable_.resize(start + npages, p.get());
  void* data = p->data;
  pmap_[data] = std::move(p);
  return data;
}

void* VirtualMemory::GetAddr(addr_t phy_addr) {  
  addr_t addr_v = phy_addr;// - fm_addr_ini;  // offset from fm_addr_init
  addr_t loc = addr_v >> kPageBits;
  CHECK_LT(loc, ptable_.size()) << "phy_addr = " << loc << "> ptable_.size() = "
                                << ptable_.size();
  Page* p = ptable_[loc];
  CHECK(p != nullptr);
  size_t offset = (loc - p->ptable_begin) << kPageBits;
  offset += addr_v & (kPageSize - 1);
  return reinterpret_cast<char*>(p->data) + offset * 64;
}

void* VirtualMemory::GetBaseAddr() {
  return fm;
}

void VirtualMemory::Write(addr_t phy_addr, void* data, size_t size) {
  void* dst = GetAddr(phy_addr);
  std::memcpy(dst, data, size);
}

void VirtualMemory::SaveNumpy(addr_t offset, int row, int col, int fraclen, std::string fname) {
  void* dst = GetAddr(offset);
  fmap_data_type *data = reinterpret_cast<fmap_data_type*>(dst);
  int trow = row / 32;
  int tcol = col / 32;
  float* df = new float[row*col];
  int idx = 0;
  for (int p = 0; p < trow; p++) {
    for (int i = 0; i < 32; i++) {
      for (int k = 0; k < tcol; k++) {
        for (int j = 0; j < 32; j++) {
          fmap_data_type value = data[k * row*32 + p*32*32 + i * 32 + j];
          if (FMAP_DATA_WIDTH == 16) {
            value = ((value & 0xFF) << 8) | ((value & 0xFF00) >> 8);
          }
          df[idx] = (double)value / ((int64_t)1 << fraclen);
          idx++;
        }
      }
    }
  }

  std::string odir = "./data/";
  struct stat ut = {0};
  if (stat(odir.c_str(), &ut) == -1) {
      mkdir(odir.c_str(), 0777);
  }
  std::vector<size_t> shape = {(size_t)row, (size_t)col};
  cnpy::npy_save(odir + fname + ".npy", df, shape);
  std::cout << "save " << odir + fname + ".npy" << "\n";
}