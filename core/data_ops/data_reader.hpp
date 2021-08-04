#ifndef data_reader_h
#define data_reader_h

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include "../utils/utils.hpp"

bool* get_bits(char c);

bool** get_cifar10_images(int image_num, std::string& data_root);
bool* get_cs1(int labels_count, int image_num, std::string& data_root);
bool** get_labels(int labels_count, int image_num, std::string& data_root);

template<typename T>
std::vector<std::vector<T>> read_sparse_arrays(int s, const std::string& data_root, int max_num = -1)
{
    std::string input = data_root + "/sparse_arrays/arr_" + std::to_string(s) + ".csv";
    std::ifstream file(input);
    std::string line;
    std::vector<std::vector<T>> result;
    for (int num_line = 0; std::getline(file, line); num_line++)
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<T> arr(s);
        int index = 0;
        while (std::getline(ss, token, ','))
        {
            //std::cout << token;
            T num = (T) std::stoi(token);
            arr[index] = num;
            index++;
        }
        result.push_back(arr);
        if (max_num != -1 && max_num == result.size())
            break;
    }
    file.close();
    return result;
}

template<typename T>
bool* read_sparse_arrays_bool(std::vector<std::vector<T>>& indices, int max_num, uint L)
{
    bool* result = (bool*) malloc(L * max_num * sizeof(bool));

    for (int i = 0; i < max_num; i++)
    {
        auto inds = indices[i];
        for (int j = 0; j < L; j++)
            result[i*L + j] = false;
        for (auto ind: inds)
            result[i*L + ind] = true;
    }
    return result;
}

template<typename T>
T* read_mask_indices(int K, int L, int N, const std::string& data_root)
{
    T* res = (T*) malloc(K * N * sizeof(T));

    std::string input = data_root + "/masks/indices_addr_" + std::to_string(L) +
            "_N_" + std::to_string(N) +
            "_K_" + std::to_string(K) + ".csv";
    std::ifstream file(input);
    std::string line;
    int index = 0;
    for (int num_line = 0; std::getline(file, line); num_line++)
    {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            //std::cout << token;
            T num = (T) std::stoi(token);
            res[index] = num;
            index++;
        }
    }
    file.close();
    return res;
}

template<typename T>
T* read_mask_indices_s(int K, int L, int N, const std::string& data_root, int s)
{
    T* res = (T*) malloc(K * N * sizeof(T));

    std::string input = data_root + "/masks/indices_addr_" + std::to_string(L) +
                        "_N_" + std::to_string(N) +
                        "_K_" + std::to_string(K) +
                        "_s_" + std::to_string(s) +
                        ".csv";
    std::ifstream file(input);
    std::string line;
    int index = 0;
    for (int num_line = 0; std::getline(file, line); num_line++)
    {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            //std::cout << token;
            T num = (T) std::stoi(token);
            res[index] = num;
            index++;
        }
    }

    file.close();
    return res;
}

#endif // !data_reader_h
