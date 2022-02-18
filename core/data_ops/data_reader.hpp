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
bool* get_cs1_txt(int labels_count, int image_num, std::string& data_root);

template<typename T>
std::vector<std::vector<T>> read_sparse_arrays(int s, const std::string& data_root, int max_num = -1)
{
    std::string input = data_root + "/sparse_arrays/arr_" + std::to_string(s) + ".csv";
    std::cout << "Reading sparse arrays from " << input << std::endl;
    std::ifstream file(input);
    std::string line;
    std::vector<std::vector<T>> result;
    int num_line = 0;
    for (num_line = 0; std::getline(file, line); num_line++)
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
    long long c = ((long long) L) * max_num * sizeof(bool);
    bool* result = (bool*) malloc(c);
    int sz = indices.size();

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
    std::cout << "Reading mask indices from " << input << std::endl;
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
                        //"_s_" + std::to_string(s) +
                        ".csv";
    std::cout << "Reading mask indices from " << input << std::endl;
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
char* read_mask_indices_s_binary(T K, T L, T N, const std::string& data_root, int s)
{
    std::string input = data_root + "/masks/indices_addr_" + std::to_string(L) +
                        "_N_" + std::to_string(N) +
                        "_K_" + std::to_string(K) +
                        //"_s_" + std::to_string(s) +
                        ".b";
    std::cout << "Reading mask indices from " << input << std::endl;
    const char* path = &input[0];

    FILE* fp = fopen(path, "rb");

    char* buffer = (char*)malloc(L * N * sizeof(char) / 8);
    fread(buffer, 1, L * N / 8, fp);
    fclose(fp);

    return buffer;
}

template<typename T>
bool* read_mask_indices_s_bool(int K, int L, int N, const std::string& data_root, int s)
{
    long long len = (long long) N * L;
    bool* result = (bool*) malloc(len * sizeof(bool));
    for (long long i = 0; i < len; i++)
    {
        result[i] = true;
    }

    std::string input = data_root + "/masks/indices_addr_" + std::to_string(L) +
                        "_N_" + std::to_string(N) +
                        "_K_" + std::to_string(K) +
                        //"_s_" + std::to_string(s) +
                        ".csv";
    std::cout << "Reading mask indices from " << input << std::endl;
    std::ifstream file(input);
    std::string line;
    int index = 0;
    for (long long num_line = 0; std::getline(file, line); num_line++)
    {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            //std::cout << token;
            T num = (T) std::stoi(token);
            long long i = num_line*L + num;
            result[i] = false;
            index++;
        }
    }

    file.close();
    return result;
}

#endif // !data_reader_h
