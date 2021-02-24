#ifndef kernels_cuh
#define kernels_cuh

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <memory>
#include <time.h>
#include <vector>
#include <Windows.h>

#include "../utils/utils.hpp"


template<typename cell_type, typename index_type>
__global__
void init_jaeckel(cell_type* cells, index_type* indices, bool* bits, uint K, uint L, uint M, uint N, int thread_count)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
	curandState state;
	curand_init(thread_num, 0, 0, &state);

	for (uint i = thread_num; i < N; i += thread_count)
	{
		for (uint j = 0; j < K; j++)
		{
			indices[i*K + j] = (index_type)(L * curand_uniform(&state));
			long rand = L * curand_uniform(&state);
			bits[i*K + j] = rand % 2;
		}

		for (uint j = 0; j < M + 1; j++)
		{
			cells[i*(M + 1) + j] = 0;
		}
	}
}


template<typename cell_type, typename index_type>
__global__
void init_jaeckel_weighted(cell_type* cells, index_type* indices, bool* bits, uint K, uint L, uint M, uint N, int thread_count)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(thread_num, 0, 0, &state);

    for (uint i = thread_num; i < N; i += thread_count)
    {
        for (uint j = 0; j < K; j++)
        {
            index_type quartet = (index_type)(L * curand_uniform(&state)/4);
            index_type el_rand = (index_type) 10*curand_uniform(&state);
            index_type el = 0;
            if (el_rand == 6 || el_rand == 7 || el_rand == 8 || el_rand == 9)
                el = 0;
            if (el_rand == 3 || el_rand == 4 || el_rand == 5)
                el = 1;
            if (el_rand == 1 || el_rand  == 2)
                el = 2;
            if (el_rand == 0)
                el = 3;

            index_type index = 4*quartet + el;
            indices[i*K + j] = index;
            long rand = L * curand_uniform(&state);
            bool bit = rand % 2;
            bits[i*K + j] = bit;
            //printf("%d %d %d %d %d %d %d %d\n", i, j, quartet, el_rand, el, index, rand, bit);
        }

        for (uint j = 0; j < M + 1; j++)
        {
            cells[i*(M + 1) + j] = 0;
        }
    }
}

template<typename cell_type, typename index_type>
__global__
void init_labels(cell_type* cells, index_type* indices, bool* bits, uint K, uint L, uint M, uint N, int thread_count)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

	for (uint i = thread_num; i < N; i += thread_count)
	{
		for (uint j = 0; j < K; j++)
		{
			bits[i * K + j] = 1;
		}

		for (uint j = 0; j < M + 1; j++)
		{
			cells[i * (M + 1) + j] = 0;
		}
	}
	if (thread_num != 0)
		return;

	int index = 0;
	for (int i = 0; i < M - 1; i++)
	{
		for (int j = i + 1; j < M; j++)
		{
			indices[index] = (index_type) i;
			indices[index + 1] = (index_type) j;
			index += 2;
		}
	}
}

template<typename index_type>
__device__
bool is_activated(index_type* indices, bool* bits, int i, uint K, bool* destination_address)
{
	for (int j = 0; j < K; j++)
	{
		int index = indices[i*K + j];

		bool equal = destination_address[index] == bits[index];
		if (!equal)
		{
			return false;
		}
	}
	return true;
}


template<typename cell_type, typename index_type, typename summation_type>
__global__
void get_activated_cells(index_type* indices, bool* bits, uint K, uint M, uint N,
	int thread_count, bool* destination_address, int* activated_indices, int* counter)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = thread_num; i < N; i += thread_count)
	{
		bool activated = is_activated(indices, bits, i, K, destination_address);
		if (activated)
		{
			int old = atomicAdd(&counter[0], 1);
			activated_indices[old] = i;
		}
	}
}

template<typename cell_type, typename index_type, typename summation_type>
__global__
void read_jaeckel(cell_type* cells, int* activated_indices, uint M, int thread_count, summation_type* sum, int activated_cells_number)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_num < M)
	{
		for (int i = 0; i < activated_cells_number; i++)
		{
			int activated_index = activated_indices[i];
			for (int j = thread_num; j < M; j += thread_count)
			{
				sum[j] += cells[activated_index * (M + 1) + j];
			}
		}
		//printf("%d %d\n", thread_num, sum[thread_num]);
	}
}

template<typename summation_type>
__global__
void get_result(summation_type* sum, bool* result, uint M, int thread_count, double threshold = 0.0)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

	for (uint i = thread_num; i < M; i += thread_count)
	{
		result[i] = sum[i] > threshold;
	}
}

template<typename cell_type, typename index_type>
__global__
void write_jaeckel(cell_type* cells, uint M, int thread_count, cell_type* to_add, int* activated_indices, int activated_cells_number)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_num < M + 1)
	{
		for (int i = 0; i < activated_cells_number; i++)
		{
			int cell_index = activated_indices[i];
			for (int j = thread_num; j < M + 1; j += thread_count)
			{
				long ind = cell_index * (M + 1) + j;
				cells[ind] = cells[ind] + to_add[j];
			}
		}
	}
}

template<typename cell_type>
__global__
void get_array_to_add(bool* value, cell_type* to_add, uint M, int times, int thread_count)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_num == 0)
		to_add[M] = times;

	for (uint i = thread_num; i < M; i += thread_count)
	{
		to_add[i] = value[i] ? times : -times;
	}
}

template<typename T>
__global__
void copy_chunk(T* data, T* chunk, int start, int chunk_size, int thread_count)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = thread_num; i < chunk_size; i += thread_count)
	{
		chunk[i] = data[start + i];
	}
}

template<typename T, typename V>
__global__
void sum_array(T* arr, int arr_len, V* sum_arr)
{
	int idx = threadIdx.x;
	V sum = 0;
	for (int i = idx; i < arr_len; i += blockDim.x)
		sum += arr[i];
	__shared__ int r[512];
	r[idx] = sum;
	__syncthreads();
	for (int size = blockDim.x / 2; size > 0; size /= 2) {
		if (idx < size)
			r[idx] += r[idx + size];
		__syncthreads();
	}
	if (idx == 0)
		*sum_arr = r[0];
}

template<typename cell_type>
__global__
void get_thresholds(cell_type* cells, int* cell_indices, int indices_count,
	double p0, double p1, uint M, double* thresholds, int thread_count)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread_num; i < indices_count; i += thread_count)
	{
		int cell_index = cell_indices[i];
		thresholds[i] = cells[cell_index *(M+1) + M] * (p1 - p0);
	}
}

template<typename cell_type>
__global__
void get_bio_decisions(cell_type* cells, bool* decisions, double* thresholds,
	int* cuda_activation_indices, int activated_cells_number, uint M, int thread_count)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < activated_cells_number; i++)
	{
		int cell_index = cuda_activation_indices[i];
		double threshold = thresholds[i];
		for (int j = thread_num; j < M; j += thread_count)
		{
			decisions[i * M + j] = cells[cell_index*(M + 1) + j] > threshold;
		}
	}
}

template<typename T>
__global__
void get_bio_result(bool* decisions, bool* result, int min_ones_count, uint M, int activated_cells_number, int thread_count)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = thread_num; i < M; i += thread_count)
	{
		int ones_count = 0;
		for (int j = 0; j < activated_cells_number; j++)
		{
			ones_count += decisions[j * M + i];
		}
		result[i] = ones_count >= min_ones_count;
	}
}

template <typename T>
__global__
void generate_small_random_matrix(int rows, int columns, T* matrix)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_num >= columns)
        return;

    curandState state;
    curand_init(thread_num, 0, 0, &state);

    for (int i = 0; i < rows; i++)
    {
        double rand = curand_uniform(&state);
        matrix[thread_num*rows + i] = rand > 0.5;
    }
}

template <typename T>
__global__
void mult_matrix(bool* left, bool* right, T* result, int left_rows, int left_columns, int right_columns,
                 int thread_count)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_num; i < left_rows*right_columns; i += thread_count)
    {
        T tmp_sum = 0;
        int left_row = i / right_columns;
        int right_column = i % right_columns;
        for (int j = 0; j < left_columns; j++)
        {
            int left_index = left_row*left_columns+j;
            int right_index = right_column*left_columns+j;
            T left_item = left[left_index];
            T right_item = right[right_index];
            T to_add = left_item * right_item;
            tmp_sum += to_add;
        }
        result[i] = tmp_sum;
    }
}

template <typename T>
__global__
void mult_matrix_xor(bool* left, bool* right, bool* result, int left_rows, int left_columns, int right_columns,
                 int thread_count)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_num; i < left_rows*right_columns; i += thread_count)
    {
        bool tmp_sum = 0;
        int left_row = i / right_columns;
        int right_column = i % right_columns;
        for (int j = 0; j < left_columns; j++)
        {
            int left_index = left_row*left_columns+j;
            int right_index = right_column*left_columns+j;
            bool left_item = left[left_index];
            bool right_item = right[right_index];
            bool to_add = left_item ^ right_item;
            tmp_sum ^= to_add;
        }
        result[i] = tmp_sum;
    }
}

template<typename T>
void check_errors(const std::string& prefix = "")
{
    auto last_error = cudaGetLastError();
    if (last_error != cudaSuccess)
    {
        throw std::exception(cudaGetErrorString(last_error));
    }
}
#endif // !kernels_cuh
