#ifndef data_reader_h
#define data_reader_h

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "../utils/utils.hpp"

bool* get_bits(char c);

bool** get_cifar10_images(int image_num, std::string& data_root);

bool** get_labels(int labels_count, int image_num, std::string& data_root);

#endif // !data_reader_h
