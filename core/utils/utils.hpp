#ifndef utils_h
#define utils_h

#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>


long get_current_time_millis();


typedef unsigned int uint;
//typedef unsigned long long ulong;

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

template <typename T>
void out(T* array, int length, char sep)
{
    for (int i = 0; i < length; i++)
    {
        std::cout << array[i] << sep;
    }
}

template <typename T>
T* noise_ones(const T* array, int length, int seed = 0)
{
    T* result = (T*) malloc(length * sizeof(T));

    std::vector<int> one_indices;
    for (int i = 0; i < length; i++)
    {
        T val = array[i];
        if (val == 1)
            one_indices.push_back(i);
        result[i] = array[i];
    }

    std::random_device rd;
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> u_distribution(0, one_indices.size());

    int swap_swap_index = u_distribution(generator);
    int swap_index = one_indices[swap_swap_index];

    result[swap_index] ^= 1;

    return result;
}
#endif // !utils_h