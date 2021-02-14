#ifndef data_reader_cpp
#define data_reader_cpp

#include <iostream>
#include "data_reader.hpp"
#include <sys/stat.h>


bool* get_bits(char c)
{
	bool* bits = (bool*)malloc(8 * sizeof(bool));
	for (int j = 0; j < 8; j++)
	{
		bits[8 - j - 1] = (c >> j) & 1;
	}
	return bits;
}

bool** get_cifar10_images(int image_num)
{
	const int image_size = 3072;
	const int pixels = image_size / 3;
	const int rows = 32;
	const int row_size = pixels / rows;
	const int columns = 32;
	const int step = image_size + 1;

	char* path = "D:\\PhD\\Code\\sdm\\sdm-cuda\\resources\\cifar10.bin";

	FILE* fp = fopen(path, "rb");

	char* buffer = (char*)malloc(step * image_num * sizeof(char));
	fread(buffer, 1, image_num*step, fp);

	bool** images = (bool**)malloc(image_num * sizeof(bool*));

	for (int i = 0; i < image_num; i++)
	{
		images[i] = (bool*)malloc(pixels * 24 * sizeof(bool));

		for (int j = 0; j < pixels; j++)
		{
			char r = buffer[1 + i*step + j];
			char g = buffer[1 + i*step + j + 1024];
			char b = buffer[1 + i*step + j + 1024 + 1024];

			bool* r_bits = get_bits(r);
			bool* g_bits = get_bits(g);
			bool* b_bits = get_bits(b);

			for (int l = 0; l < 8; l++)
			{
				images[i][j * 24 + l] = r_bits[l];
				images[i][j * 24 + l + 8] = g_bits[l];
				images[i][j * 24 + l + 16] = b_bits[l];
			}

			free(r_bits);
			free(g_bits);
			free(b_bits);
		}
	}
	free(buffer);
	return images;
}

bool** get_labels(int labels_count, int image_num)
{
	int label_size_bits = labels_count + 8 - (labels_count % 8);
	int label_size_bytes = label_size_bits / 8;

	bool** labels = (bool**)malloc(image_num * sizeof(bool*));

	char* buffer = (char*)malloc(label_size_bytes * image_num * sizeof(char));

	char* path = "C:\\Development\\PhD\\sdm-plots\\resources\\labels4.bin";
	FILE* fp = fopen(path, "rb");
	fread(buffer, 1, image_num * label_size_bytes, fp);
	int a = 2 + 3;

	for (int i = 0; i < image_num; i++)
	{
		labels[i] = (bool*)malloc(label_size_bytes * 8 * sizeof(bool));
		for (int j = 0; j < label_size_bytes; j++)
		{
			bool* bits = get_bits(buffer[i * label_size_bytes + j]);
			for (int k = 0; k < 8; k++)
			{
				labels[i][j * 8 + k] = bits[k];
			}
		}
	}
	return labels;
}


#endif // !data_reader_cpp
