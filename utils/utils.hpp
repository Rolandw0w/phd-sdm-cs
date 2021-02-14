#ifndef utils_h
#define utils_h

#include <math.h>
#include <random>


typedef unsigned int uint;

enum ComputingDevice {
	CPU,
	GPU
};

// Distance functions
uint hamming_distance(bool *x, bool *y, uint dim);

// Address functions
bool* strict_address_detector(bool *value, uint dim);

uint from_bits(bool *bits, uint bits_num);

bool* to_bits(int num, uint dim);

bool* noise(const bool* value, const uint length, double probability);
bool* noise(const bool* value, const uint length, const uint error_num);


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