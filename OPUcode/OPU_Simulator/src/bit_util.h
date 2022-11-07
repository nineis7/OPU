#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#ifndef FSIM_BIT_UTIL_H_
#define FSIM_BIT_UTIL_H_

#define REVERSE_BYTES16(value) (((value & 0xFF) << 8) | ((value & 0xFF00) >> 8))
#define REVERSE_BYTES32(value) (((value & 0xFF) << 24) | ((value & 0xFF00) << 8) | ((value & 0xFF0000) >> 8) | ((value & 0xFF000000) >> 24))

inline sim_data_type Saturate(sim_data_type x, bool positive, int wl) {
  sim_data_type mask = ~((sim_data_type)0) << (wl - 1);
  if (x == 0) {
    return x;
  } else if (!positive && ((mask & x) != mask)) {
    return mask;  // negative max, keep negative when extended to 32 bit int
  } else if (positive && (mask & x)) {
    return ~mask;
  }
  return x;
}

inline void Saturate(std::vector<sim_data_type>& data, int wl) {
  for (auto &x : data) {
    x = Saturate(x, x > 0, wl);
  }
}

template <typename T, int W>
void writeOut(std::ofstream& os,
  std::vector<T> data, bool dump = false, bool debug = false) {  // W - #bit to print
  if (!dump) return;
  // mask
  sim_data_type mask = 0;
  for (int i = 0; i < W; i++) {
    mask |= ((sim_data_type)1 << i);
  }
  // hex
  std::stringstream ss;
  int i = 0;
  for (auto item : data) {
    std::stringstream sstream;
    sim_data_type value = static_cast<sim_data_type>(item);
    sstream << std::hex << (value & mask);
    std::string ret = sstream.str();
    /*if (debug) {
      std::stringstream sh;
      sh << std::hex << mask;
      std::cout << "mask : " << sh.str() << "\n";
      std::stringstream sv;
      sv << std::hex << static_cast<sim_data_type>(item);
      std::cout << "value : " << sv.str() << "\n";
      std::cout << ret << "\n";
    }*/
    int len = ret.length() * 4;
    /*if (len > W) {
      std::stringstream sss;
      sss << std::hex << value; 
      std::cout << "value : " << sss.str() << "\n";
      std::stringstream sh;
      sh << std::hex << mask;
      std::cout << "mask : " << sh.str() << "\n";
      std::cout << ret << "\n";
      std::cout << "#bit : " << len << " v.s. " << W << "\n";
      assert(0);
      ret = ret.substr(len - W, W);
    } else*/ if (len < W) {
      std::string prefix;
      for (int ii = 0; ii + 4 <= W - len; ii+=4) {
        prefix = (value >= 0) ? "0" : "f";
        ret = prefix + ret;
      }
      len = ret.length() * 4;
      if (W - len == 3) {
        prefix = (value >= 0) ? "0" : "7";
        ret = prefix + ret;
      } else if (W - len == 2) {
        prefix = (value >= 0) ? "0" : "3";
        ret = prefix + ret;
      } else if (W - len == 1) {
        prefix = (value >= 0) ? "0" : "1";
        ret = prefix + ret;
      }
    }
    ss << ret << " ";
    /*if (debug) {
      std::cout << ret << "\n";
    }
    i++;
    if (debug && i == 5) exit(1);*/
  }
  os << ss.str() << "\n";
}
#endif  // FSIM_BIT_UTIL_H_
