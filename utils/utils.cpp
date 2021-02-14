#ifndef utils_cpp
#define utils_cpp

#include <iostream>
#include <memory>
#include <vector>

#include "utils.h"


uint hamming_distance(bool *x, bool *y, uint dim)
{
	uint dist = 0;
	for (uint d = 0; d < dim; d++)
	{
		bool xd = x[d];
		bool yd = y[d];
		dist += x[d] ^ y[d];
	}
	return dist;
}

bool* strict_address_detector(bool *value, uint dim)
{
	bool* address = (bool*)malloc(dim * sizeof(bool));
	for (uint i = 0; i < dim; i++)
	{
		address[i] = value[i];
	}
	return address;
}

uint from_bits(bool *bits, uint bits_num)
{
	uint number = 0;
	for (int i = bits_num - 1; i >= 0; i--)
	{
		bool bit = bits[i];
		if (bit)
			number += (1 << (bits_num - 1 - i));
	}
	return number;
}

bool* to_bits(int num, uint dim)
{
	bool* bits = (bool*)malloc(dim * sizeof(bool));
	for (uint i = 0; i < dim; i++)
	{
		bits[dim - 1 - i] = (num >> i) & 1;
	}
	return bits;
}

bool* noise(const bool* value, const uint length, const double probability)
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::bernoulli_distribution distribution(probability);

	bool* noised_value = (bool*)malloc(length * sizeof(bool));

	for (uint i = 0; i < length; i++)
	{
		auto rand = distribution(generator);
		noised_value[i] = rand ^ value[i];
	}
	return noised_value;
}

bool* noise(const bool* value, const uint length, const uint error_num)
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::bernoulli_distribution b_distribution(0.5);
	std::uniform_int_distribution<int> u_distribution(0, length);

	bool* noised_value = (bool*)malloc(length * sizeof(bool));
	memcpy(noised_value, value, length * sizeof(bool));

	for (uint i = 0; i < error_num; i++)
	{
		bool rand = b_distribution(generator);
		if (rand)
		{
			int index = u_distribution(generator);
			noised_value[index] = rand ^ noised_value[index];
		}
	}
	return noised_value;
}

#endif // !utils_cpp
