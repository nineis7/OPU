#include "bert_activations.h"

/////////////////////////////////////////
// Pieceweise Linear Approximations 
/////////////////////////////////////////

// Pwlin EXP (Scalar)
void pwlin_exp(int32_t data, int16_t& out,
               vector<int16_t>& x0,   // (16,9)
               vector<int16_t>& y0,   // (16,9)
               vector<int16_t>& k0) { // (16,9)

	out = 0;
	for (int i=4; i>=0; i--) {
		if (data > x0[i]) {
			int32_t sub = data - x0[i]; // (17,9) - (16,9) = (17, 9)
			int32_t mult = sub * ((int32_t)k0[i]); // (32, 18)
			int64_t add = mult + (((int32_t)y0[i])<<9); // (32, 18) + (16, 9) -> (32, 18) 
			out = (int16_t)(add>>4); // shift to get (32, 14) and truncate to (16, 14)
			break;
		}
	}
}

// Pwlin EXP (Vector)
void pwlin_exp_array(vector<int32_t> array, vector<int16_t>& out, bool verbose) {

	vector<int16_t> x0{-3194, -1536, -1024, -512, 0};
	vector<int16_t> y0{1, 25, 69, 188, 512};
	vector<int16_t> k0{8, 44, 119, 324, 324};

	if (verbose) {
		cout << "X0";
		print_hex_array(x0);
		cout << "Y0";
		print_hex_array(y0);
		cout << "K0";
		print_hex_array(k0);
		cout << endl;
	}

	for (it_t i=0; i<array.size(); i++) {
		pwlin_exp(array[i], out[i], x0, y0, k0);
	}
}

/////////////////////////////////////////
// Main Nonlinearity Functions
/////////////////////////////////////////

// Softmax
vector<int16_t> deep_softmax(vector<int16_t> data, bool verbose) {

	// Input
	if (verbose) {
		cout << "Input (16, 9)" << endl;
		print_fp_array(data, 9);
		print_hex_array(data);
		cout << endl;
	}

	// Max
	int16_t max_val = *max_element(data.begin(), data.end());
	if (verbose) {
		cout << "Max val (16, 9): " << endl;
		print_fp(max_val, pow(2, 9));
		cout << endl;
		print_hex(max_val);
		cout << endl << endl;
	}

	// X-Xmax
	vector<int32_t> sub(data.begin(), data.end());
	for (it_t i=0; i<sub.size(); i++) {
		sub[i] = ((int32_t)data[i]) - ((int32_t)max_val);
	}
	if (verbose) {
		cout << "Subtraction (17, 9):" << endl;
		print_fp_array(sub, 9);
		print_hex_array(sub);
		cout << endl;
	}

	// Exp
	vector<int16_t> exp(sub.begin(), sub.end());
	pwlin_exp_array(sub, exp, verbose);
	if (verbose) {
		cout << "Exponent (16, 14):" << endl;
		print_fp_array(exp, 14);
		print_hex_array(exp);
		cout << endl;
	}

	// Sum
	int32_t sum_val = accumulate(exp.begin(), exp.end(), (int32_t)0);
	if (verbose) {
		cout << "Sum (32, 14):" << endl;
		print_fp(sum_val, 14);
		cout << endl;
		print_hex(sum_val);
		cout << endl << endl;
	}

	// Divide
	vector<int16_t> div(exp.begin(), exp.end());
	for (it_t i=0; i<div.size(); i++) {
		div[i] = (((int64_t)div[i])<<14) / ((int32_t)sum_val);
	}
	if (verbose) {
		cout << "Division (16, 14):" << endl;
		print_fp_array(div, 14);
		print_hex_array(div);
		cout << endl;
	}

	return div;
}

// Layer Norm
vector<int16_t> deep_layer_norm(vector<int16_t> data, bool verbose) {

	// Input
	if (verbose) {
		cout << "Input (16, 6)" << endl;
		print_fp_array(data, 6);
		print_hex_array(data);
		cout << endl;
	}

	// Mean
	int32_t sum_val = accumulate(data.begin(), data.end(), (int32_t)0);
	int32_t mean_val = sum_val * (1024/data.size());
	if (verbose) {
		cout << "Mean (26, 16):" << endl;
		print_fp(mean_val, pow(2,16));
		cout << endl;
		print_hex(mean_val);
		cout << endl << endl;
	}

	// X-Xmean
	vector<int32_t> f1(data.begin(), data.end());
	for (it_t i=0; i<f1.size(); i++) {
		f1[i] = (((int32_t)data[i])<<10) - ((int32_t)mean_val);
	}
	if (verbose) {
		cout << "X-X_mean (27, 16):" << endl;
		print_fp_array(f1, 16);
		print_hex_array(f1);
		cout << endl;
	}

	// Variance
	int64_t var = 0;
	for (it_t i=0; i<f1.size(); i++) {
		var += (((int64_t)f1[i])*((int64_t)f1[i]));
	}
	var /= f1.size();
	if (verbose) {
		cout << "Var (54, 32):" << endl;
		print_fp(var, pow(2,32));
		cout << endl;
		print_hex(var);
		cout << endl << endl;
	}

	// Variance reduced precision for cordic
	int64_t max_threshold = pow(2, 47)-1;
	int64_t min_threshold = -pow(2, 47);
	if (var > max_threshold) {
		var = max_threshold;
	}
	if (var < min_threshold) {
		var = min_threshold;
	}
	if (verbose) {
		cout << "Var reduced precision for cordic input (48, 32):" << endl;
		print_fp(var, pow(2,32));
		cout << endl;
		print_hex(var);
		cout << endl << endl;
	}

	// Square root(var) - Cordic
	double f2_double = sqrt_cordic((double)var, 48);
	int64_t f2 = (int64_t)(f2_double*pow(2,24));
	if (verbose) {
		cout << "Stdev = sqrt(var) (48, 40):" << endl;
		print_fp(f2, pow(2,40));
		cout << endl;
		print_hex(f2);
		cout << endl << endl;
	}

	// Normalized output ((X-mean)/f2) = f1/f2
	vector<int64_t> normalized_output(f1.begin(), f1.end());
	for (it_t i=0; i<normalized_output.size(); i++) {
		double normalized_output_double = (((double)f1[i])*pow(2,24)) / ((double)f2);
		normalized_output[i] = (int64_t)(normalized_output_double * pow(2,40));
	}
	if (verbose) {
		cout << "Normalized output (48, 40):" << endl;
		print_fp_array(normalized_output, 40);
		print_hex_array(normalized_output);
		cout << endl;
	}

	// 16b output 
	vector<int16_t> output(f1.begin(), f1.end());
	for (it_t i=0; i<output.size(); i++) {
		output[i] = (int16_t)(normalized_output[i] / pow(2,32));
	}
	if (verbose) {
		cout << "16b output (16, 8):" << endl;
		print_fp_array(output, 8);
		print_hex_array(output);
		cout << endl;
	}

	return output;
}