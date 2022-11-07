#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
#include "cordic.h"

using namespace std;
using it_t = unsigned long;

/////////////////////////////////////////
// Helper Functions for Printing
/////////////////////////////////////////

// Print array as floating point
template <typename T>
void print_fp(T data, float scale_factor) {
	cout << data/scale_factor << " ";
}
template <typename T>
void print_fp_array(vector<T> array, int num_frac_bits) {
	cout << "[";
	float scale_factor = pow(2, num_frac_bits);
	for (it_t i=0; i<array.size(); i++) {
		print_fp(array[i], scale_factor);
	}
	cout << "]" << endl;
}

// Print array as hex
template <typename T>
void print_hex(T data) {
	cout << "0x" << hex << data << dec << " ";
}
template <typename T>
void print_hex_array(vector<T> array) {
	cout << "[";
	for (it_t i=0; i<array.size(); i++) {
		print_hex(array[i]);
	}
	cout << "]" << endl;
}

/////////////////////////////////////////
// Pieceweise Linear Approximations 
/////////////////////////////////////////

void pwlin_exp      (int32_t data, int16_t& out, vector<int16_t>& x0,
                     vector<int16_t>& y0, vector<int16_t>& k0);
void pwlin_exp_array(vector<int32_t> array, vector<int16_t>& out, bool verbose=false);

/////////////////////////////////////////
// Main Nonlinearity Functions
/////////////////////////////////////////

vector<int16_t> deep_softmax   (vector<int16_t> data, bool verbose=false);
vector<int16_t> deep_layer_norm(vector<int16_t> data, bool verbose=false);
