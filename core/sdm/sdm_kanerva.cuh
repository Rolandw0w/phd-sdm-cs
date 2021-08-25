#ifndef sdm_kanerva_cuh
#define sdm_kanerva_cuh

#include "sdm_base.cuh"


template<typename cell_type, typename index_type, typename summation_type>
struct SDM_KANERVA : SDM_BASE<cell_type, index_type, summation_type>
{
public:
    uint d;
    uint L;
    uint M;
    uint N;
    double p0;

    cell_type* cells;
    bool* addresses;

    uint block_count;
    uint threads_per_block;
    uint thread_count;

    SDM_KANERVA(uint d, uint L, uint M, uint N, uint block_count, uint threads_per_block, double p0);
    ~SDM_KANERVA();

    bool* read(const bool* value);
    bool* read(const bool* value, int iter_count);
    bool* read(const bool* value, const bool* address);

    uint write(const bool *value);
    uint write(const bool *value, const int times);
    uint write(const bool *value, const bool *address);
    uint write(const bool *value, const bool *address, const int times);

    void print_state(uint cells_num = 10);
};

template<typename cell_type, typename index_type, typename summation_type>
SDM_KANERVA<cell_type, index_type, summation_type>::SDM_KANERVA(uint d, uint L, uint M, uint N, uint block_count,
                                                                uint threads_per_block, double p0)
{
    this->d = d;
    this->L = L;
    this->M = M;
    this->N = N;
    this->d = d;
    this->p0 = p0;
    this->block_count = block_count;
    this->threads_per_block = threads_per_block;

    thread_count = this->block_count * this->threads_per_block;

    cuda_malloc(&cells, N * (M + 1));
    cuda_malloc(&addresses, M * N);

    kernel_decorator(
            init_kanerva<cell_type>,
            block_count, threads_per_block, true,
            cells, addresses, L, M, N, thread_count, this->p0
    );
}

template<typename cell_type, typename index_type, typename summation_type>
SDM_KANERVA<cell_type, index_type, summation_type>::~SDM_KANERVA()
{
    cuda_free(cells);
    cuda_free(addresses);
}

template<typename cell_type, typename index_type, typename summation_type>
bool* SDM_KANERVA<cell_type, index_type, summation_type>::read(const bool* value, const bool* address)
{
    bool* cuda_value;
    cuda_malloc(&cuda_value, M);
    cuda_memcpy_to_gpu(cuda_value, value, M);

    int* cuda_activation_indices;
    cuda_malloc(&cuda_activation_indices, N);

    summation_type* cuda_sum;
    cuda_malloc(&cuda_sum, M);
    cuda_memset(cuda_sum, 0, M);

    bool* cuda_result;
    cuda_malloc(&cuda_result, M);

    bool* cuda_address;
    cuda_malloc(&cuda_address, L);
    cuda_memcpy_to_gpu(cuda_address, address, L);

    int* cuda_activation_counter;
    cuda_malloc(&cuda_activation_counter, 1);

    int* activation_counter = (int*)malloc(sizeof(int));
    activation_counter[0] = 0;
    cuda_memcpy_to_gpu(cuda_activation_counter, activation_counter, 1);

    kernel_decorator(
            get_activated_cells_kanerva<bool>,
            block_count, threads_per_block, true,
            addresses, L, N, d, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter
    );

    cuda_memcpy_from_gpu(activation_counter, cuda_activation_counter, 1);
    int activated_cells_number = activation_counter[0];
    //std::cout << activated_cells_number << std::endl;

    if (activated_cells_number == 0)
    {
        bool* result = (bool*)malloc(M * sizeof(bool));
        memset(result, 0, M * sizeof(bool));

        free(activation_counter);

        cuda_free(cuda_activation_counter);
        cuda_free(cuda_address);
        cuda_free(cuda_activation_indices);
        cuda_free(cuda_sum);
        cuda_free(cuda_value);
        cuda_free(cuda_result);

        return result;
    }

    kernel_decorator(
            read_jaeckel<cell_type, index_type, summation_type>,
            block_count, threads_per_block, true,
            cells, cuda_activation_indices, M, thread_count, cuda_sum, activated_cells_number
    );

    kernel_decorator(
            get_result<summation_type>,
            block_count, threads_per_block, true,
            cuda_sum, cuda_result, M, thread_count, 0.0
    );

    cuda_free(cuda_activation_counter);
    cuda_memset(cuda_sum, 0, M);

    bool* result = (bool*)malloc(M * sizeof(bool));
    cuda_memcpy_from_gpu(result, cuda_result, M);

    free(activation_counter);

    cuda_free(cuda_address);
    cuda_free(cuda_activation_indices);
    cuda_free(cuda_result);
    cuda_free(cuda_sum);
    cuda_free(cuda_value);

    return result;
}

template<typename cell_type, typename index_type, typename summation_type>
bool* SDM_KANERVA<cell_type, index_type, summation_type>::read(const bool* value)
{
    return this->read(value, value);
}

template<typename cell_type, typename index_type, typename summation_type>
bool* SDM_KANERVA<cell_type, index_type, summation_type>::read(const bool* value, const int iter_num)
{
    bool* cuda_value;
    cuda_malloc(&cuda_value, M);
    cuda_memcpy_to_gpu(cuda_value, value, M);

    int* cuda_activation_indices;
    cuda_malloc(&cuda_activation_indices, N);

    summation_type* cuda_sum;
    cuda_malloc(&cuda_sum, M);
    cuda_memset(cuda_sum, 0, M);

    for (int i = 0; i < iter_num; i++)
    {
        int* cuda_activation_counter;
        cuda_malloc(&cuda_activation_counter, 1);

        int* activation_counter = (int*)malloc(sizeof(int));
        activation_counter[0] = 0;
        cuda_memcpy_to_gpu(cuda_activation_counter, activation_counter, 1);

        kernel_decorator(
                get_activated_cells_kanerva<bool>,
                block_count, threads_per_block, true,
                addresses, L, N, d, thread_count, cuda_value, cuda_activation_indices, cuda_activation_counter
        );

        cuda_memcpy_from_gpu(activation_counter, cuda_activation_counter, 1);
        int activated_cells_number = activation_counter[0];

        kernel_decorator(
                read_jaeckel<cell_type, index_type, summation_type>,
                block_count, threads_per_block, true,
                cells, cuda_activation_indices, M, thread_count, cuda_sum, activated_cells_number
        );

        summation_type* sum = (summation_type*)malloc(M*sizeof(summation_type));
        cuda_memcpy_from_gpu(sum, cuda_sum, M);

        kernel_decorator(
                get_result<summation_type>,
                block_count, threads_per_block, true,
                cuda_sum, cuda_value, M, thread_count, 0.0
        );

        cuda_free(cuda_activation_counter);
        cuda_memset(cuda_sum, 0, M);

        free(activation_counter);
    }
    bool* result = (bool*)malloc(M * sizeof(bool));
    cuda_memcpy_from_gpu(result, cuda_value, M);

    cuda_free(cuda_activation_indices);
    cuda_free(cuda_sum);
    cuda_free(cuda_value);

    return result;
}

template<typename cell_type, typename index_type, typename summation_type>
uint SDM_KANERVA<cell_type, index_type, summation_type>::write(const bool *value)
{
    return this->write(value, 1);
}

template<typename cell_type, typename index_type, typename summation_type>
uint SDM_KANERVA<cell_type, index_type, summation_type>::write(const bool *value, const int times)
{
    return this->write(value, value, times);
}

template<typename cell_type, typename index_type, typename summation_type>
uint SDM_KANERVA<cell_type, index_type, summation_type>::write(const bool *value, const bool *address)
{
    return this->write(value, address, 1);
}

template<typename cell_type, typename index_type, typename summation_type>
uint SDM_KANERVA<cell_type, index_type, summation_type>::write(const bool *value, const bool *address, const int times)
{
    bool* cuda_value;
    cuda_malloc(&cuda_value, M);
    cuda_memcpy_to_gpu(cuda_value, value, M);

    bool* cuda_address;
    cudaMalloc((void **)&cuda_address, L);
    cuda_memcpy_to_gpu(cuda_address, address, L);

    cell_type* cuda_to_add;
    cuda_malloc(&cuda_to_add, M + 1);

    kernel_decorator(
            get_array_to_add<cell_type>,
            block_count, threads_per_block, true,
            cuda_value, cuda_to_add, M, times, thread_count
    );

    int* cuda_activation_indices;
    cuda_malloc(&cuda_activation_indices, N);

    int* cuda_activation_counter;
    cuda_malloc(&cuda_activation_counter, 1);

    int* activation_counter = (int*)malloc(sizeof(int));
    activation_counter[0] = 0;
    cuda_memcpy_to_gpu(cuda_activation_counter, activation_counter, 1);

    kernel_decorator(
            get_activated_cells_kanerva<bool>,
            block_count, threads_per_block, true,
            addresses, L, N, d, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter
    );

    cuda_memcpy_from_gpu(activation_counter, cuda_activation_counter, 1);
    int activated_cells_number = activation_counter[0];

    if (activated_cells_number == 0)
    {
        free(activation_counter);

        cuda_free(cuda_activation_counter);
        cuda_free(cuda_activation_indices);
        cuda_free(cuda_address);
        cuda_free(cuda_to_add);
        cuda_free(cuda_value);
        return 0;
    }

    kernel_decorator(
            write_jaeckel<cell_type, index_type>,
            block_count, threads_per_block, true,
            cells, M, thread_count, cuda_to_add, cuda_activation_indices, activated_cells_number
    );

    free(activation_counter);

    cuda_free(cuda_activation_counter);
    cuda_free(cuda_activation_indices);
    cuda_free(cuda_address);
    cuda_free(cuda_to_add);
    cuda_free(cuda_value);

    return activated_cells_number;
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_KANERVA<cell_type, index_type, summation_type>::print_state(uint cells_num)
{
    cell_type* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));
    bool* address = (bool*)malloc(L * sizeof(bool));

    cell_type* cuda_cell;
    cuda_malloc(&cuda_cell, M + 1);

    bool* cuda_address;
    cuda_malloc(&cuda_address, L);

    for (uint i = 0; i < cells_num; i++)
    {
        std::cout << "addr_" << i << ": ";
        kernel_decorator(
                copy_chunk<bool>,
                block_count, threads_per_block, true,
                addresses, cuda_address, i*L, L, thread_count
        );
        cuda_memcpy_from_gpu(address, cuda_address, L);

        for (int l = 0; l < L; l++)
        {
            std::cout << address[l] << ",";
        }
        std::cout << std::endl;

        std::cout << "cell_" << i << ": ";

        kernel_decorator(
                copy_chunk<cell_type>,
                block_count, threads_per_block, true,
                cells, cuda_cell, i*(M + 1), (M + 1), thread_count
        );
        cuda_memcpy_from_gpu(cell, cuda_cell, M + 1);
        for (uint j = 0; j < M; j++)
        {
            std::cout << cell[j] << ",";
        }
        std::cout << " acts=" << cell[M];
        std::cout << std::endl;
    }
    cuda_free(cuda_cell);
    cuda_free(cuda_address);

    free(cell);
    free(address);
}
#endif // !sdm_kanerva_cuh
