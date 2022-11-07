#ifndef FSIM_DEBUG_CONFIG_H_
#define FSIM_DEBUG_CONFIG_H_

#define DEBUG_PREFIX "debug_"

//#define DEBUG_TXT_OUTPUT
#ifdef DEBUG_TXT_OUTPUT

#define DEBUG_DMA_FM
#define DEBUG_DMA_FM_FILENAME DEBUG_PREFIX "dma_fm.txt"

#define DEBUG_DMA_KER
#define DEBUG_DMA_KER_FILENAME DEBUG_PREFIX "dma_ker.txt"

#define DEBUG_OUT_ADDER_A
#define DEBUG_OUT_ADDER_A_FILENAME DEBUG_PREFIX "adder_a.txt"

#define DEBUG_OUT_ADDER_B
#define DEBUG_OUT_ADDER_B_FILENAME DEBUG_PREFIX "adder_b.txt"

#define DEBUG_PSUM
#define DEBUG_PSUM_FILENAME DEBUG_PREFIX "psum.txt"

#define DEBUG_PSUM_CUT
#define DEBUG_PSUM_CUT_FILENAME DEBUG_PREFIX "psum_cut.txt"

#endif

#endif
