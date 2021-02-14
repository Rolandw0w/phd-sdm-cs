#ifndef data_reader_h
#define data_reader_h

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

bool* get_bits(char c);

bool** get_cifar10_images(int image_num);

bool** get_labels(int labels_count, int image_num);

#endif // !data_reader_h
