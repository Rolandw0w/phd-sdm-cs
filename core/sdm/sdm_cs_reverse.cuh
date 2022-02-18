#ifndef sdm_cs_reverse_cuh
#define sdm_cs_reverse_cuh

#include <thread>

#include "sdm_base.cuh"
#include "../utils/utils.hpp"


template<typename cell_type, typename index_type, typename summation_type, typename value_type>
struct SDM_CS_REVERSE// : SDM_BASE<cell_type, index_type, summation_type>
{
public:
    uint K;
    uint L;
    uint M;
    uint N;

    cell_type* cells;
    index_type* indices;
    bool* bits;

    uint block_count;
    uint threads_per_block;
    uint thread_count;

    SDM_CS_REVERSE(uint K, uint L, uint M, uint N, uint block_count, uint threads_per_block, index_type* mask_indices = NULL);
    ~SDM_CS_REVERSE();

    double* read(const bool* address);

    int write(const value_type* value);
    int write(const value_type* value, const bool* address);
    int write(const value_type* value, const bool* address, double mult);

    std::vector<std::vector<cell_type>> get_whisker_boxes();
};

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
SDM_CS_REVERSE<cell_type, index_type, summation_type, value_type>::SDM_CS_REVERSE(uint K, uint L, uint M, uint N, uint block_count, uint threads_per_block, index_type* mask_indices)
{
    this->K = K;
    this->L = L;
    this->M = M;
    this->N = N;
    this->block_count = block_count;
    this->threads_per_block = threads_per_block;

    thread_count = this->block_count * this->threads_per_block;

    long long N_ = N;

    cuda_malloc(&cells, N_ * (M + 1));
    cuda_malloc(&indices, N_ * L / 8);
    ///cuda_malloc(&bits, N_ * K);

    kernel_decorator(
            init_cs_sdm<cell_type>,
            block_count, threads_per_block, true,
            cells, M, N, thread_count
    );
    if (mask_indices != NULL)
        cuda_memcpy_to_gpu(indices, mask_indices, N_ * L / 8);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
SDM_CS_REVERSE<cell_type, index_type, summation_type, value_type>::~SDM_CS_REVERSE()
{
    cuda_free(cells);
    cuda_free(indices);
    //cuda_free(bits);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
double* SDM_CS_REVERSE<cell_type, index_type, summation_type, value_type>::read(const bool* address)
{
    bool* cuda_address;
    cuda_malloc(&cuda_address, L);
    cuda_memcpy_to_gpu(cuda_address, address, L);

    int* cuda_activation_indices;
    cuda_malloc(&cuda_activation_indices, N);

    double* cuda_sum;
    cuda_malloc(&cuda_sum, M);
    cuda_memset(cuda_sum, 0, M);

    int* cuda_activation_counter;
    cuda_malloc(&cuda_activation_counter, 1);

    int* activation_counter = (int*)malloc(sizeof(int));
    activation_counter[0] = 0;
    cuda_memcpy_to_gpu(cuda_activation_counter, activation_counter, 1);

    kernel_decorator(
            get_activated_cells_reverse<index_type>,
            block_count, threads_per_block, true,
            indices, K, L, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter
    );

    cuda_memcpy_from_gpu(activation_counter, cuda_activation_counter, 1);

//    int* activation_indices = (int*) malloc(activation_counter[0] * sizeof(int));
//    cuda_memcpy_from_gpu(activation_indices, cuda_activation_indices, activation_counter[0] * sizeof(int));
//    std::cout << std::endl;
//    for (int i = 0; i < activation_counter[0]; i++)
//        std::cout << activation_indices[i] << " ";
//    std::cout << std::endl;

    if (activation_counter[0] == 0)
    {
        double* result = (double*) malloc(M * sizeof(double));
        memset(result, 0, M * sizeof(double));

        free(activation_counter);

        cuda_free(cuda_activation_counter);
        cuda_free(cuda_activation_indices);
        cuda_free(cuda_sum);
        cuda_free(cuda_address);

        return result;
    }
//    kernel_decorator(
//            get_acts_sum<cell_type, double>,
//            block_count, threads_per_block, true,
//            cells, cuda_activation_indices, M, cuda_activation_counter, cuda_sum_act, thread_count
//    );
//    double sum_activated = sum_act[0];

    kernel_decorator(
            read_cs1_v4<cell_type, index_type, summation_type>,
            block_count, threads_per_block, true,
            cells, cuda_activation_indices, M, thread_count, cuda_sum, cuda_activation_counter
    );

    double* result = (double*) malloc(M * sizeof(double));
    cuda_memcpy_from_gpu(result, cuda_sum, M);

    for (int i = 0; i < M; i++)
        result[i] /= activation_counter[0];

    free(activation_counter);
//    free(sum_act);

    cuda_free(cuda_activation_counter);
    cuda_free(cuda_activation_indices);
    cuda_free(cuda_sum);
    cuda_free(cuda_address);
    //cuda_free(cuda_sum_act);

    return result;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
int SDM_CS_REVERSE<cell_type, index_type, summation_type, value_type>::write(const value_type *value)
{
    return this->write(value, value);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
int SDM_CS_REVERSE<cell_type, index_type, summation_type, value_type>::write(const value_type *value, const bool *address)
{
    return this->write(value, address, 1.0);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
int SDM_CS_REVERSE<cell_type, index_type, summation_type, value_type>::write(const value_type *value, const bool *address, double mult)
{
    value_type* cuda_value;
    cuda_malloc(&cuda_value, M);
    cuda_memcpy_to_gpu(cuda_value, value, M);

    bool* cuda_address;
    cuda_malloc(&cuda_address, L);
    cuda_memcpy_to_gpu(cuda_address, address, L);

    int* cuda_activation_indices;
    cuda_malloc(&cuda_activation_indices, N);

    int* cuda_activation_counter;
    cuda_malloc(&cuda_activation_counter, 1);

    int* activation_counter = (int*) malloc(sizeof(int));
    activation_counter[0] = 0;
    cuda_memcpy_to_gpu(cuda_activation_counter, activation_counter, 1);

    kernel_decorator(
            get_activated_cells_reverse<index_type>,
            block_count, threads_per_block, true,
            indices, K, L, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter
    );

    cuda_memcpy_from_gpu(activation_counter, cuda_activation_counter, 1);

    int activated_cells_number = activation_counter[0];

    if (activated_cells_number == 0)
    {
        cuda_free(cuda_activation_counter);
        cuda_free(cuda_activation_indices);
        cuda_free(cuda_address);
        cuda_free(cuda_value);

        free(activation_counter);

        return 0;
    }
    //std::cout<<activated_cells_number<<",";

    kernel_decorator(
            write_cs1_v2<cell_type, index_type, value_type>,
            block_count, threads_per_block, true,
            cells, M, thread_count, cuda_value, cuda_activation_indices, activated_cells_number, mult
    );

    cuda_free(cuda_activation_counter);
    cuda_free(cuda_activation_indices);
    cuda_free(cuda_address);
    cuda_free(cuda_value);

    free(activation_counter);

    return activated_cells_number;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
std::vector<std::vector<cell_type>> SDM_CS_REVERSE<cell_type, index_type, summation_type, value_type>::get_whisker_boxes()
{
    long long size = (long long) N * (M + 1);
    auto host_cells = (cell_type*) malloc(size * sizeof(cell_type));
    cuda_memcpy_from_gpu(host_cells, cells, size);

    std::vector<std::vector<cell_type>> result(M + 1);

    std::vector<std::thread> threads;
    threads.reserve((M + 1));

    std::vector<std::vector<cell_type>> arrs(M + 1);

    for (int i = 0; i < M + 1; i++)
    {
        std::vector<cell_type> arr(N);
        for (int j = 0; j < N; j++)
            arr[j] = host_cells[i + j*(M+1)];
        arrs[i] = arr;
        std::thread thread_obj(SortClass<std::vector<cell_type>>(), std::ref(arrs[i]));
        threads.push_back(std::move(thread_obj));
    }
    free(host_cells);

    for (int i = 0; i < M + 1; i++)
        threads[i].join();

    for (int i = 0; i < M + 1; i++)
    {
        auto w_box = whisker_box_sorted(arrs[i]);
        result[i] = w_box;
    }
    return result;
}

#endif // !sdm_cs_reverse_cuh
