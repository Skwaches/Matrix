#pragma once
#include <vector>
#include <functional>
#include <iostream>
#include "matrix.cuh"

// Whether or not to use the GPU for operations
#ifdef CUDA_AVAILABLE 
extern const bool GPU;
#else
#define GPU false
#endif

class Matrix {
	private:
	public:
		int rows, columns;
		std::vector<float> data;

		Matrix (int rows=1, int columns=1);
		Matrix (std::initializer_list<std::initializer_list<float>>elements);
		
		auto begin()const{ return data.begin();}
		auto end()const{ return data.end();}
		size_t size()const{return data.size();};

		float determinant();
		float& operator[](const int);
		float  operator[](const int)const;
		bool operator==(const Matrix& other)const;


		// Pass all the elements through a function and assign the element the output.
		void apply(std::function<float(float)> func);

		// Replace all the elements by the output of a function.
		void replace(std::function<float(void)> func);

		// Pass all the elements through a function and assign the output to a new matrix.
		// Leaving the original unchanged	
		Matrix modify(std::function<float(float)> func);

		// Pass all the elements through a function and do nothing.
		template<typename func>
			void pass(func fn){
				for (auto element: data)
					fn(element);
			}

		Matrix& operator*=(const Matrix& other);
		Matrix& operator+=(const Matrix& other);
		Matrix& operator-=(const Matrix& other);

		Matrix operator*(const Matrix& other) const;
		Matrix operator+(const Matrix& other) const;
		Matrix operator-(const Matrix& other) const;


		Matrix& operator*=(const float other);
		Matrix& operator/=(const float other);
		Matrix& operator+=(const float other);
		Matrix& operator-=(const float other);

		Matrix operator*(const float other) const;
		Matrix operator/(const float other) const;
		Matrix operator+(const float other) const;
		Matrix operator-(const float other) const;

		friend std::ostream& operator << (std::ostream& os, const Matrix& A);
};
