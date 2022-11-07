 #ifndef FSIM_NONLINEAR_H
 #define FSIM_NONLINEAR_H

#include <vector>
#include <cmath>

#include "smem.h"
#include "bert_activations.h"
#include "logging.h"

/*
 * Activation units
 */
enum ActivationType {
  BypassActivation,
  ReLU,
  pReLU,
  GeLU
};

template<int parallelism>
class ActivationUnit {
 public:
  int units = parallelism;
  float scale_factor = 1.0;
  void Run(std::vector<sim_data_type>& data, ActivationType type) {
    for (auto& item : data) {
      if (type == ActivationType::BypassActivation) {

      } else if (type == ActivationType::ReLU) {
        item = (item < 0)? 0 : item;
      } else if (type == ActivationType::pReLU) {
        // leaky_relu: *0.125 == >>3 and round
        if (item < 0) {
          if (item & 0x4) {
            item = (item >> 3) + 1;
          } else {
            item = item >> 3;
          }
        }
      } else if (type == ActivationType::GeLU) {
        std::cout << "GeLU Not Implemented\n";
        exit(1);
      }
    }
  }
};

enum PoolingType {
  BypassPooling,
  Max,
  Average
};

template<int parallelism>
class PoolingUnit {
 public:
  int units = parallelism;
  void Run(std::vector<sim_data_type>& data_o, std::vector<sim_data_type> data_i, 
    PoolingType type, bool st, bool ed, int window_size);
};

enum ElwOpcode {
  BypassElw,
  Add,
  Mul,
  Sub,
  Div
};

template<int parallelsim>
class ElementwiseALU {
 public:
  int units = parallelsim;
  void Run(std::vector<sim_data_type>& a, std::vector<sim_data_type>& b, ElwOpcode opcode) {
    // Unroll
    for (int i = 0; i < units; i++) {
      if (opcode == ElwOpcode::BypassElw) {
        
      } else if (opcode == ElwOpcode::Add) {
        a[i] += b[i];
      } else if (opcode == ElwOpcode::Sub) {
        a[i] -= b[i];
      } else if (opcode == ElwOpcode::Mul) {
        a[i] *= b[i];
      } else if (opcode == ElwOpcode::Div) {
        a[i] /= b[i];
      }
    }
  }
};

enum NonlinearOpcode {
  BypassNonlinear,
  LayerNorm,
  Softmax,
  Tanh,
  FloatSoftmax,
  FloatLayerNorm,
  FloatGeLU
};

template<int parallelism, int bandwidth_per_bank, int depth_per_register_bank>
class NVM {
 public:
  static const int banks = parallelism;  // one bank for one row
  using vreg_t = SRAM<bandwidth_per_bank, depth_per_register_bank*banks, 1>;
  vreg_t vregs_;

  // simulation only
  int input_fraclen;
  int output_fraclen;
  int beta_fraclen;
  int gamma_fraclen;
  int rows;

  NVM() {
    if (FMAP_DATA_WIDTH != 16) {
      std::cout << "[ERROR] NVM only supports 16-bit\n";
      //exit(1);
    }
  }

  uint32_t GetBankAddr(int bank_idx, int offset_in_bank) {
    return bank_idx * depth_per_register_bank + offset_in_bank;
  }

  template<typename T>
  void Run(std::vector<T>& data_i, NonlinearOpcode opcode) {
    std::vector<int16_t> data_i_16b;
    for (auto &x : data_i) {
      data_i_16b.push_back(static_cast<int16_t>(x));
    }
    std::vector<int16_t> data_o_16b;
    if (opcode == NonlinearOpcode::BypassNonlinear) {
      data_o_16b = data_i_16b;
    } else if (opcode == NonlinearOpcode::LayerNorm) {
      data_o_16b = deep_layer_norm(data_i_16b, false);
    } else if (opcode == NonlinearOpcode::Softmax) {
      //data_o_16b = deep_softmax(data_i_16b, false);

      // exp(x) = sinh(x) + cosh(x)
      // where cosh(x), sinh(x) = cordic(cosh/sinh)(x)
      // input x is 3QN format (i.e., 1 sign bit, 2 int bits, fl=13)
      // outputs are both 2QN format (fl=14)
      // exp=Q1.2.14
      std::vector<double> t;
      assert(input_fraclen <= 13);
      assert(output_fraclen >= 14);
      for (auto &x : data_i_16b) {
        sim_data_type value = Saturate(x << (13 - input_fraclen), x > 0, 16) >> (13 - input_fraclen);
        double tmp = (double)value / (1<<2);
        tmp = std::exp(tmp);
        tmp = std::min(tmp, (double)(4 - (double)1 / std::pow(2, 13)));
        tmp = std::max(tmp, (double)(-4));
        t.push_back(tmp);
      }
      double sum = 0;
      for (auto &x : t) sum += x;
      for (auto &x : t) x /= sum;
      for (auto &x : t) {
        data_o_16b.push_back((int16_t)(x * ((int64_t)1 << (output_fraclen - 14))));
      }
    } else if (opcode == NonlinearOpcode::FloatSoftmax) {
      std::vector<double> t;
      for (auto &x : data_i_16b) {
        t.push_back(std::exp((double)x / ((int64_t)1 << input_fraclen)));
      }
      double sum = 0;
      for (auto &x : t) sum += x;
      for (auto &x : t) x /= sum;
      for (auto &x : t) {
        //data_o_16b.push_back((int16_t)(x * ((int64_t)1 << output_fraclen)));
        // we saturate output data here to handle the corner case below
        // word length = 16, fraction length = 15, data = 0x1
        // (int16_t)data << 15 = -1 without saturation!!!
        sim_data_type xs = (sim_data_type)(x * ((int64_t)1 << output_fraclen));
        data_o_16b.push_back((int16_t)Saturate(xs, xs>0, 16));
      }
    } else if (opcode == NonlinearOpcode::FloatLayerNorm) {
      // fetch beta
      std::vector<double> beta;
      for (int k = 0; k < rows; k++) {
        std::vector<sim_data_type> data = vregs_.AsVec(GetBankAddr(0, k), DRAM_BANDWIDTH/FMAP_DATA_WIDTH, FMAP_DATA_WIDTH);
        for (auto &x : data) {
          beta.push_back((double)x / ((int64_t)1 << beta_fraclen));
        }
      }
      // fetch gamma
      std::vector<double> gamma;
      for (int k = 0; k < rows; k++) {
        std::vector<sim_data_type> data = vregs_.AsVec(GetBankAddr(1, k), DRAM_BANDWIDTH/FMAP_DATA_WIDTH, FMAP_DATA_WIDTH);
        for (auto &x : data) {
          gamma.push_back((double)x / ((int64_t)1 << gamma_fraclen));
        }
      }
      // input
      std::vector<double> t;
      for (auto &x : data_i_16b) {
        t.push_back((double)x / ((int64_t)1 << input_fraclen));
      }
      int n = (int)t.size();
      // mean
      double sum = 0;
      for (auto &x : t) sum += x;
      double mean = sum / n;
      // variance
      double var = 0;
      for (auto &x : t) var += (x - mean) * (x - mean);
      var /= n;
      // normalize
      for (auto &x : t) {
        x = (x - mean) / std::sqrt(var + std::pow(10,-12));
      }
      //std::cout << "n : " << n << "\n";
      //std::cout << "rows : " << rows << "\n";
      //std::cout << "sum : " << sum << "\n";
      //std::cout << "mean : " << mean << "\n";
      //std::cout << "var : " << var << "\n";
      // scale and bias
      for (int i = 0; i < n; i++) {
        t[i] = t[i] * gamma[i] + beta[i]; 
      }
      //for (auto &x : t) std::cout << x << " ";std::cout << "\n\n";exit(1);
      // output
      for (auto &x : t) {
        data_o_16b.push_back((int16_t)(x * ((int64_t)1 << output_fraclen)));
      }
    } else if (opcode == NonlinearOpcode::FloatGeLU) {
      // 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      std::vector<double> t;
      for (auto &x : data_i_16b) {
        t.push_back((double)x / ((int64_t)1 << input_fraclen));
      }
      for (auto &x : t) {
        x = 0.5 * x * (1 + std::tanh(std::sqrt(2 / 3.14155269) * (x + 0.044715 * std::pow(x, 3))));
      }
      for (auto &x : t) {
        data_o_16b.push_back((int16_t)(x * ((int64_t)1 << output_fraclen)));
      }
    }
    data_i.clear();
    for (auto &x : data_o_16b) {
      data_i.push_back(static_cast<T>(x));
    }
  }
};

 #endif
