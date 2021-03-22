#ifndef utils_h
#define utils_h

#include <cmath>
#include <random>
#include <stdexcept>


typedef unsigned int uint;
typedef unsigned long long ulong;

// Distance functions
uint hamming_distance(bool *x, bool *y, uint dim);

// Address functions
bool* strict_address_detector(const bool *value, uint dim);

uint from_bits(const bool *bits, uint bits_num);

bool* to_bits(int num, uint dim);

bool* noise(const bool* value, uint length, double probability);
bool* noise(const bool* value, uint length, uint error_num);

short* noise(const short* value, uint length, uint error_num);
short* noise(const short* value, uint length, uint error_num, int seed);


template<typename T>
void sort(T* arr, int size)
{
	for (int i = 0; i < size - 1; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			if (arr[i] > arr[j])
			{
				T temp = arr[i];
				arr[i] = arr[j];
				arr[j] = temp;
			}
		}
	}
}

template<typename T>
T median(T* arr, int size)
{
	sort(arr, size);
	if (size % 2 == 0)
	{
		return (arr[size / 2] + arr[size / 2 - 1]) / 2;
	}
	return arr[size / 2];
}

template<typename T>
T sum(T* arr, int size)
{
	T s = 0;
	for (int i = 0; i < size; i++)
		s += arr[i];
	return s;
}

template<typename T>
T mean(T* arr, int size)
{
	return sum(arr, size) / size;
}
#endif // !utils_h