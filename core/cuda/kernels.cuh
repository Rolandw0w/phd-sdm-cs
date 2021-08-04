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
//#include <Windows.h>

#include "functions.cuh"
//#include "../utils/utils.hpp"


template<typename cell_type, typename index_type>
__global__
void init_jaeckel(cell_type* cells, index_type* indices, bool* bits, uint K, uint L, uint M, uint N, int thread_count,
                  double p0 = 0.5)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
	curandState state;
	curand_init(thread_num, 0, 0, &state);

	for (uint i = thread_num; i < N; i += thread_count)
	{
		for (uint j = 0; j < K; j++)
		{
			indices[i*K + j] = (index_type)(L * curand_uniform(&state));
			double rand = curand_uniform(&state);
			bits[i*K + j] = rand > p0;
			//printf("%d %d %d\n", i*K + j, indices[i*K + j], bits[i*K + j]);
		}

		for (uint j = 0; j < M + 1; j++)
		{
			cells[i*(M + 1) + j] = 0;
		}
	}
}


template<typename cell_type, typename index_type>
__global__
void init_jaeckel_num_ones(cell_type* cells, index_type* indices, bool* bits, uint K, uint L, uint M, uint N, int thread_count,
                           index_type* num_ones, double p0 = 0.5)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(thread_num, 0, 0, &state);

    for (uint i = thread_num; i < N; i += thread_count)
    {
        for (uint j = 0; j < K; j++)
        {
            indices[i*K + j] = (index_type)(L * curand_uniform(&state));
            double rand = curand_uniform(&state);
            bool bit = rand > p0;
            bits[i*K + j] = bit;
            num_ones[i] += bit;
            //printf("%d %d %d\n", i*K + j, indices[i*K + j], bits[i*K + j]);
        }
        //printf("%d=%d,", i, num_ones[i]);

        for (uint j = 0; j < M + 1; j++)
        {
            cells[i*(M + 1) + j] = 0;
        }
    }
}


template<typename cell_type, typename index_type>
__global__
void init_jaeckel_ones(cell_type* cells, index_type* indices, bool* bits, uint K, uint L, uint M, uint N, int thread_count)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(thread_num, 0, 0, &state);

    for (uint i = thread_num; i < N; i += thread_count)
    {
        for (uint j = 0; j < K; j++)
        {
            indices[i*K + j] = (index_type)(L * curand_uniform(&state));
            bits[i*K + j] = 1;
            //printf("%d %d %d\n", i*K + j, indices[i*K + j], bits[i*K + j]);
        }

        for (uint j = 0; j < M + 1; j++)
        {
            cells[i*(M + 1) + j] = 0;
        }
    }
}


template<typename cell_type, typename index_type>
__global__
void init_jaeckel_zeroes(cell_type* cells, index_type* indices, bool* bits, uint K, uint L, uint M, uint N, int thread_count)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(thread_num, 0, 0, &state);

    for (uint i = thread_num; i < N; i += thread_count)
    {
        for (uint j = 0; j < K; j++)
        {
            indices[i*K + j] = (index_type)(L * curand_uniform(&state));
            bits[i*K + j] = 0;
            //printf("%d %d %d\n", i*K + j, indices[i*K + j], bits[i*K + j]);
        }

        for (uint j = 0; j < M + 1; j++)
        {
            cells[i*(M + 1) + j] = 0;
        }
    }
}


template<typename cell_type>
__global__
void init_kanerva(cell_type* cells, bool* addresses, uint L, uint M, uint N, int thread_count, double p0 = 0.5)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(thread_num, 0, 0, &state);

    for (uint i = thread_num; i < N; i += thread_count)
    {
        for (uint j = 0; j < M + 1; j++)
        {
            cells[i*(M + 1) + j] = 0;
        }
        for (uint j = 0; j < L; j++)
        {
            double rand = curand_uniform(&state);
            addresses[i*L + j] = rand > p0;
        }
    }
}


template<typename cell_type, typename index_type>
__global__
void init_cs1(cell_type* cells, index_type* indices, uint K, uint L, uint M, uint N, int thread_count)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(thread_num, 0, 0, &state);

    for (uint i = thread_num; i < N; i += thread_count)
    {
        for (uint j = 0; j < K; j++)
        {
            index_type rand = L * curand_uniform(&state);
            index_type ind = 2*rand - (L - 1);
            indices[i*K + j] = ind;
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
void init_labels(cell_type* cells, index_type* indices, uint K, uint L, uint M, uint N, int thread_count)
{
	int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(thread_num, 0, 0, &state);

	for (long i = thread_num; i < N; i += thread_count)
	{
		for (uint j = 0; j < M + 1; j++)
		{
			cells[i * (M + 1) + j] = 0;
		}
		for (uint k = 0; k < K; k++)
        {
            indices[K*i+k] = (index_type)(L * curand_uniform(&state));
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

template<typename index_type>
__device__
bool is_activated_bool(index_type* indices, bool flag, int i, uint K, bool* destination_address)
{
    for (int j = 0; j < K; j++)
    {
        int index = indices[i*K + j];

        bool equal = destination_address[index] == flag;
        if (!equal)
        {
            return false;
        }
    }
    return true;
}

template<typename index_type, typename value_type>
__device__
bool is_activated_cs1(index_type* indices, bool* bits, int i, uint K, value_type* destination_address)
{
    for (int j = 0; j < K; j++)
    {
        int index = indices[i*K + j];
        bool bit = bits[index];

        value_type el = destination_address[index];
        bool flag = (bit && (el > 0)) ||  (!bit && (el < 0));
        if (!flag)
        {
            return false;
        }
    }
    return true;
}


template<typename cell_type, typename index_type, typename summation_type, typename value_type>
__global__
void get_activated_cells_cs1(index_type* indices, bool* bits, uint K, uint M, uint N,
                             int thread_count, value_type* destination_address, int* activated_indices, int* counter)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = thread_num; i < N; i += thread_count)
    {
        bool activated = is_activated_cs1(indices, bits, i, K, destination_address);
        if (activated)
        {
            int old = atomicAdd(&counter[0], 1);
            activated_indices[old] = i;
        }
    }
}


template<typename cell_type, typename index_type, typename summation_type>
__global__
void get_activated_cells_bool(index_type* indices, bool flag, uint K, uint M, uint N,
                              int thread_count, bool* destination_address, int* activated_indices, int* counter)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = thread_num; i < N; i += thread_count)
    {
        bool activated = is_activated_bool(indices, flag, i, K, destination_address);
        if (activated)
        {
            int old = atomicAdd(&counter[0], 1);
            activated_indices[old] = i;
        }
    }
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

template<typename dist_type>
__device__
bool is_activated_kanerva(bool* addresses, bool* destination_address, uint i, uint L, uint d)
{
    dist_type dist = 0;
    for (int l = 0; l < L; l++)
    {
        bool flag = addresses[i*L + l];
        bool index_dist = flag ^ destination_address[l];
        dist += index_dist;
        if (dist > d)
            return false;
    }
    return true;
}


template<typename index_type>
__global__
void get_activated_cells_kanerva(bool* addresses, uint L, uint N, uint d,
                           int thread_count, bool* destination_address, int* activated_indices, int* counter)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint i = thread_num; i < N; i += thread_count)
    {
        bool activated = is_activated_kanerva<uint>(addresses, destination_address, i, L, d);
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

template <typename cell_type, typename sum_type>
__global__
void get_acts_sum(cell_type* cells, int* activated_indices, uint M, int* activation_counter, sum_type* sum_act, int thread_count)
{
    int activated_cells_number = activation_counter[0];
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

    sum_type d = 0;
    for (int i = thread_num; i  < activated_cells_number; i += thread_count)
    {
        long long activated_index = activated_indices[i];
        long long cell_index = activated_index * (M + 1) + M;
        cell_type m = cells[cell_index];
        d += m;
        //printf("\n%lld %lld %lld %d %d,", activated_index, c, cell_index, m, sum_act[0]);
    }
    atomicAdd(&sum_act[0], d);
}

template<typename cell_type, typename index_type, typename summation_type>
__global__
void read_cs1(cell_type* cells, int* activated_indices, uint M, int thread_count, double* sum, int* activation_counter, double* sum_act)
{
    int activated_cells_number = activation_counter[0];
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_num < M)
    {
        for (int i = 0; i < activated_cells_number; i++)
        {
            long long activated_index = activated_indices[i];
            long long cell_start = activated_index * (M + 1);
            for (int j = thread_num; j < M; j += thread_count)
            {
                long long cell_index = cell_start + j;
                sum[j] += cells[cell_index];
            }
        }
    }
//    __syncthreads();
//    if (thread_num == 0)
//    {
//        for (int j = 0; j < M; j++)
//        {
//            sum[j] = (abs(sum_act[0]) > 1e-6) ? sum[j] / sum_act[0] : 0.0;
//            //printf("%d %d %f %f,", activation_counter[0], j, sum[j], sum_act[0]);
//        }
//    }
}

template<typename cell_type, typename index_type, typename summation_type>
__global__
void read_cs1_v3(cell_type* cells, int* activated_indices, uint M, int thread_count, double* sum, int* activation_counter, double* sum_act)
{
    int activated_cells_number = activation_counter[0];
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_num; i < activated_cells_number; i += thread_count)
    {
        long long activated_index = activated_indices[i];
        long long cell_start = activated_index * (M + 1);
        for (int j = 0; j < M; j++)
        {
            long long cell_index = cell_start + j;
            atomicAdd(&sum[j], (double) cells[cell_index]);
        }
    }
}

template<typename cell_type, typename index_type, typename summation_type>
__global__
void read_cs1_s2(cell_type* cells, int* activated_indices, uint M, int thread_count, double* sum, int* activation_counter, int* sum_act,
                 index_type* num_ones)
{
    int activated_cells_number = activation_counter[0];
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_num < M)
    {
        for (int i = 0; i < activated_cells_number; i++)
        {
            ulong activated_index = activated_indices[i];
            index_type weight = num_ones[activated_index];
            weight = (weight == 0) ? 1 : weight;
            for (int j = thread_num; j < M; j += thread_count)
            {
                ulong cell_index = activated_index * (M + 1) + j;
                double d = ((double) weight) * cells[cell_index];
                sum[j] += d;
            }
        }
    }
    __syncthreads();
    if (thread_num == 0)
    {
        for (int j = 0; j < M; j++)
        {
            sum[j] = (abs(sum_act[0]) > 1e-6) ? sum[j] / sum_act[0] : 0.0;
            //printf("%d %d %f %f,", activation_counter[0], j, sum[j], sum_act[0]);
        }
    }
}

template<typename cell_type, typename index_type, typename summation_type>
__global__
void read_cs1_v2(cell_type* cells, int* activated_indices, uint M, int thread_count, double* sum, int* activation_counter, int* sum_act)
{
    int activated_cells_number = activation_counter[0];
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_num < M)
    {
        for (int i = thread_num; i < activated_cells_number; i+=thread_count)
        {
            ulong activated_index = activated_indices[i];
            for (int j = 0; j < M; j++)
            {
                ulong cell_index = activated_index * (M + 1) + j;
                sum[j] += cells[cell_index];
            }
        }
    }
    __syncthreads();
    if (thread_num == 0)
    {
        for (int j = 0; j < M; j++)
        {
            sum[j] /= sum_act[0];
        }
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
		//printf("%d %d %d %f\n", thread_num, sum[i], result[i], threshold);
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
				long long ind = cell_index * (M + 1) + j;
				cells[ind] = cells[ind] + to_add[j];
			}
		}
	}
}

template<typename cell_type, typename index_type, typename value_type>
__global__
void write_cs1(cell_type* cells, uint M, int thread_count, value_type* to_add, int* activated_indices, int activated_cells_number)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
//    if (thread_num < 1)
//    {
//        for (int i = 0; i < activated_cells_number; i++) {
//            int cell_index = activated_indices[i];
//            printf("%d ", cell_index);
//        }
//        printf("\n");
//    }

//    if (thread_num == 0)
//        printf("\n\nwrite activated_cells_number=%d\n\n", activated_cells_number);
    if (thread_num < M + 1)
    {
        for (int i = 0; i < activated_cells_number; i++)
        {
            ulong cell_index = activated_indices[i];
            for (int j = thread_num; j < M; j += thread_count)
            {
                ulong ind = cell_index * (M + 1) + j;
                cells[ind] = cells[ind] + to_add[j];
            }
            if (thread_num == 0)
                cells[cell_index * (M + 1) + M] += 1;
        }
    }
}

template<typename cell_type, typename index_type, typename value_type>
__global__
void write_cs1_v2(cell_type* cells, uint M, int thread_count, value_type* to_add, int* activated_indices, int activated_cells_number,
                  double mult)
{
    int thread_num = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_num; i < activated_cells_number; i += thread_count)
    {
        long long cell_index = activated_indices[i];
        for (int j = 0; j < M; j++)
        {
            long long ind = cell_index * (M + 1) + j;
            auto a = (cell_type) mult*to_add[j];
            //printf("\n%f %d %d %llu %llu %d %d %d ", mult, to_add[j], a, cell_index, ind, M, j, activated_cells_number);
            cells[ind] += a;
        }
        cells[cell_index * (M + 1) + M] += ((int) mult);
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
    __syncthreads();
    if (idx == 0)
        printf("%d\n", sum_arr[0]);
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
	if (idx == 0)
        printf("%d\n", sum_arr[0]);
}

template<typename T, typename V>
__global__
void sum_array_naive(T* arr, int arr_len, V* sum_arr)
{
    int idx = threadIdx.x;
    if (idx == 0)
        for (int i = 0; i < arr_len; i++)
            if (arr[i])
                atomicAdd(&sum_arr[0], 1);
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
            T left_item = 2*left[left_index] - 1;
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

//template<typename T>
//void check_errors(const std::string& prefix = "")
//{
//    auto last_error = cudaGetLastError();
//    if (last_error != cudaSuccess)
//    {
//        throw std::exception(cudaGetErrorString(last_error));
//    }
//}
#endif // !kernels_cuh
