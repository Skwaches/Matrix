#include "matrix.hpp"
#include <stdexcept>
#include "matrix.cuh"

bool GPU = true;
Matrix::Matrix(int rows, int columns):
	rows(rows),columns(columns),data(rows * columns,0.00)
{}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> elements){
	rows = elements.size();
	columns = rows ? elements.begin()[0].size(): 0;

	for(auto& row : elements){
		if ((int)row.size() != columns) {
			throw std::invalid_argument("All rows in the initializer list must have the same number of columns.");
		}
		data.insert(data.end(), row.begin(), row.end());
	}

}

// Add the product of positive diagonals. 
// Subtract that of negative diagonals.
// This only works for a 3x3 matrix unfortunately.
float Matrix::determinant(){
	if ( rows != columns )
		throw std::invalid_argument ( "Non-square matrix doesn't not have a defined determinant");
	if (rows != 3)
		throw std::invalid_argument ( "Only a 3x3 matrix's determinant has been set-up");
	float determinant = 0;

	for (int i = 0; i < rows; i++){ 
		// Column that you start at
		float temp1 = 1;
		float temp2 = 1;
		for ( int j = 0; j < rows; j++){ // Iterator to fill
			temp1 *= data[ ( (i+j) % rows) * rows + j];
			temp2 *= data[ (i+j)%rows * columns + (rows - 1 - j)];
		}
		determinant += temp1 - temp2;
	}
	return determinant;
}

float& Matrix::operator[](const int index){
	return data[index];
}
float Matrix::operator[](const int index)const{
	return data[index];
}

bool Matrix::operator==(const Matrix& other)const{
	return (rows == other.rows && columns == other.columns && other.data == data);
}


void Matrix::apply(std::function<float(float)> func){
	for(auto& element: data)
		element = func(element);
}
void Matrix::replace(std::function<float(void)> func){
	for(auto& element: data)
		element = func();
}
Matrix Matrix::modify(std::function<float(float)> func){
	Matrix other = *this;
	other.apply(func);
	return other;
}



Matrix& Matrix::operator*=(const float other){
	if (GPU)
		GPU_multiplication_D(data, other, data);
	else {
		for(auto& element: data)
			element *= other;
	}
	return *this;
}
Matrix& Matrix::operator/=(const float other){
	if (GPU)
		GPU_multiplication_D(data,1.0/other, data);
	else {
		for(auto& element: data)
			element /= other;
	}
	return *this;
}
Matrix& Matrix::operator+=(const float other){
	if (GPU)
		GPU_addition_D(data,other,data);
	else {
		for(auto& element: data)
			element += other;
	}
	return *this;
}
Matrix& Matrix::operator-=(const float other){
	if (GPU)
		GPU_addition_D(data, -other, data);
	else {
		for(auto& element: data)
			element -= other;
	}
	return *this;
}


Matrix Matrix::operator*(const float other) const{
	Matrix result = *this;
	result *= other;
	return result;
}
Matrix Matrix::operator/(const float other) const{
	Matrix result = *this;
	result /= other;
	return result;
}
Matrix Matrix::operator+(const float other) const{
	Matrix result = *this;
	result += other;
	return result;
}
Matrix Matrix::operator-(const float other) const{
	Matrix result = *this;
	result -= other;
	return result;
}


Matrix& Matrix::operator*=(const Matrix& other){
	if (columns != other.rows){
		throw std::invalid_argument("Matrices are not compatable for multiplication");
	}

	std::vector<float> result(rows*other.columns, 0.0);
	if (GPU){
		GPU_multiplication(data,  other.data, result, rows, other.columns, columns);
	}
	else {
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < other.columns; j++){
				for(int k = 0; k < columns; k++){
					result[i*other.columns + j] += data[i*columns + k] * other.data[j + other.columns*k];
				}
			}
		}
	}

	data = std::move(result);
	columns = other.columns;
	return *this;
}
Matrix& Matrix::operator+=(const Matrix& other){
	if( rows != other.rows || columns != other.columns)
		throw std::invalid_argument("Matrices are not compatable for addition");

	if(GPU){
		GPU_addition(data, other.data, data);
	}
	else {
		for(size_t i = 0; i < data.size(); i++)
			data[i] += other.data[i];
	}
	return *this;
}
Matrix& Matrix::operator-=(const Matrix& other){
	*this += other * -1;
	return *this;
}

Matrix Matrix::operator*(const Matrix& other) const{
	Matrix result = *this;
	result *= other;
	return result;
}
Matrix Matrix::operator+(const Matrix& other) const{
	Matrix result = *this;
	result += other;
	return result;
}
Matrix Matrix::operator-(const Matrix& other) const{
	Matrix result = *this;
	result -= other;
	return result;
}

#include <iomanip>
std::ostream& operator << (std::ostream& os, const Matrix& A){
	const int columnWidth = 10;
	const int decimals = 1;

	os << std::fixed << std::setprecision(decimals);
	for (size_t j = 0; j < A.data.size(); j++){
		if (j % A.columns == 0)
			os << "| "; 

		os << std::right << std::setw(columnWidth) << A[j];

		if ((j + 1) % A.columns == 0)
			os << "|\n";

		}
	return os;
}
