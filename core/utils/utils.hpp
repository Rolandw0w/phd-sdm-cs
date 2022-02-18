#ifndef utils_h
#define utils_h

#include <algorithm>
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

template<typename T>
bool* to_bits(T num, uint dim)
{
    bool* bits = (bool*)malloc(dim * sizeof(bool));
    for (uint i = 0; i < dim; i++)
    {
        bits[dim - 1 - i] = (num >> i) & 1;
    }
    return bits;
}

bool* noise(const bool* value, uint length, double probability);
bool* noise(const bool* value, uint length, uint error_num);

short* noise(const short* value, uint length, uint error_num);
short* noise(const short* value, uint length, uint error_num, int seed);

char* get_time();


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
T max(T* arr, int size)
{
    T m = arr[0];
    for (int i = 1; i < size; i++)
        if (m < arr[i])
            m = arr[i];
    return m;
}

template<typename T>
T sum(T* arr, int size)
{
	T s = 0;
	for (int i = 0; i < size; i++)
		s += arr[i];
	return s;
}

template<typename T, typename S>
S sum2(const T* arr, int size)
{
    S s = 0;
    for (int i = 0; i < size; i++)
        s += (S) arr[i];
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

template<typename T>
bool* get_sparse_array(std::vector<T>& ones_indices, int length)
{
    bool* sparse_array = (bool*) malloc(length * sizeof(bool));
    for (int index = 0; index < length; index++)
    {
        sparse_array[index] = false;
    }
    for (auto index: ones_indices)
    {
        sparse_array[index] = true;
    }
    return sparse_array;
}

template<typename Array>
void sort(Array& arr)
{
    std::sort(arr.begin(), arr.end());
}

template<typename Array>
class SortClass
{
public:
    void operator()(Array& arr);
};

template<typename Array>
void SortClass<Array>::operator()(Array& arr)
{
    sort(arr);
}

template<typename T>
std::vector<T> whisker_box_sorted(std::vector<T>& arr)
{
    std::vector<T> result(5);

    auto const q0 = 0;
    auto const q1 = arr.size() / 4;
    auto const q2 = arr.size() / 2;
    auto const q3 = q1 + q2;
    auto const q4 = arr.size() - 1;

    result[0] = arr[q0];
    result[1] = arr[q1];
    result[2] = arr[q2];
    result[3] = arr[q3];
    result[4] = arr[q4];

    return result;
}

template<typename T>
std::vector<double> whisker_box(std::vector<T>& arr)
{
    sort(arr);
    std::vector<double> result = whisker_box_sorted(arr);

    return result;
}

#endif // !utils_h