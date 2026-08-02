#pragma once

#include <vector>
void GPU_addition_D(
		std::vector<float>& A,
		const float B,
		std::vector<float> &C);
void GPU_multiplication_D(
		std::vector<float>& A,
		const float B,
		std::vector<float>& C
		);

//Perform size checks elsewhere!
void GPU_addition(
		std::vector<float>& A,
		const std::vector<float>& B,
		std::vector<float> &C);
void GPU_multiplication(
		std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
		int rows, // First Matrix rows
		int columns, // Second Matrix columns
		int shared // Columns of the first & Rows of the second
		);

void quitCUDA();
