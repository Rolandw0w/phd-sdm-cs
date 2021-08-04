#ifndef sdm_cs2_s1_cuh
#define sdm_cs2_s1_cuh

#include "sdm_base.cuh"


template<typename cell_type, typename index_type, typename summation_type, typename value_type>
struct SDM_CS2_S2// : SDM_BASE<cell_type, index_type, summation_type>
{
public:
    uint K;
    uint L;
    uint M;
    uint N;

    cell_type* cells;
    index_type* indices;
    bool* bits;
    index_type* num_ones;

    uint block_count;
    uint threads_per_block;
    uint thread_count;

    SDM_CS2_S2(uint K, uint L, uint M, uint N, uint block_count, uint threads_per_block);
    ~SDM_CS2_S2();

    double* read(const bool* address);

    int write(const value_type* value);
    int write(const value_type* value, const bool* address);
    int write(const value_type* value, const bool* address, double mult);

    cell_type get_min_activations();
    cell_type get_max_activations();

    long get_activations_num();

    void print_state();
};

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
SDM_CS2_S2<cell_type, index_type, summation_type, value_type>::SDM_CS2_S2(uint K, uint L, uint M, uint N, uint block_count, uint threads_per_block)
{
    this->K = K;
    this->L = L;
    this->M = M;
    this->N = N;
    this->block_count = block_count;
    this->threads_per_block = threads_per_block;

    thread_count = this->block_count * this->threads_per_block;

    cuda_malloc(&cells, N * (M + 1));
    cuda_malloc(&indices, K * N);
    cuda_malloc(&bits, K * N);
    cuda_malloc(&num_ones, N);

    kernel_decorator(
            init_jaeckel_num_ones<cell_type, index_type>,
            block_count, threads_per_block, true,
            cells, indices, bits, K, L, M, N, thread_count, num_ones, 0.5
    );
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
SDM_CS2_S2<cell_type, index_type, summation_type, value_type>::~SDM_CS2_S2()
{
    cuda_free(cells);
    cuda_free(indices);
    cuda_free(bits);
    cuda_free(num_ones);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
double* SDM_CS2_S2<cell_type, index_type, summation_type, value_type>::read(const bool* address)
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
            get_activated_cells<cell_type, index_type, summation_type>,
            block_count, threads_per_block, true,
            indices, bits, K, M, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter
    );

    cuda_memcpy_from_gpu(activation_counter, cuda_activation_counter, 1);

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

    int* cuda_sum_act;
    cuda_malloc(&cuda_sum_act, 1);
    kernel_decorator(
            get_acts_sum<cell_type, int>,
            block_count, threads_per_block, true,
            cells, cuda_activation_indices, M, cuda_activation_counter, cuda_sum_act, thread_count
    );

    kernel_decorator(
            read_cs1_s2<cell_type, index_type, summation_type>,
            block_count, threads_per_block, true,
            cells, cuda_activation_indices, M, thread_count, cuda_sum, cuda_activation_counter, cuda_sum_act, num_ones
    );

    double* result = (double*) malloc(M * sizeof(double));
    cuda_memcpy_from_gpu(result, cuda_sum, M);

    free(activation_counter);

    cuda_free(cuda_activation_counter);
    cuda_free(cuda_activation_indices);
    cuda_free(cuda_sum);
    cuda_free(cuda_address);
    cuda_free(cuda_sum_act);

    return result;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
int SDM_CS2_S2<cell_type, index_type, summation_type, value_type>::write(const value_type *value)
{
    return this->write(value, value);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
int SDM_CS2_S2<cell_type, index_type, summation_type, value_type>::write(const value_type *value, const bool *address)
{
    return this->write(value, address, 1.0);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
int SDM_CS2_S2<cell_type, index_type, summation_type, value_type>::write(const value_type *value, const bool *address, double mult)
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
            get_activated_cells<cell_type, index_type, summation_type>,
            block_count, threads_per_block, true,
            indices, bits, K, M, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter
    );

    cuda_memcpy_from_gpu(activation_counter, cuda_activation_counter, 1);

    int activated_cells_number = activation_counter[0];
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
cell_type SDM_CS2_S2<cell_type, index_type, summation_type, value_type>::get_min_activations()
{
    cell_type min = 0;

    cell_type* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));

    cell_type* cuda_cell;
    cuda_malloc(&cuda_cell, M + 1);

    for (uint i = 0; i < N; i++)
    {
        kernel_decorator(
                copy_chunk<cell_type>,
                block_count, threads_per_block, true,
                cells, cuda_cell, i*(M + 1), (M + 1), thread_count
        );
        cuda_memcpy_from_gpu(cell, cuda_cell, M + 1);

        if (min > cell[M])
        {
            min = cell[M];
        }
    }
    free(cell);
    cuda_free(cuda_cell);

    return min;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
cell_type SDM_CS2_S2<cell_type, index_type, summation_type, value_type>::get_max_activations()
{
    cell_type max = 0;

    cell_type* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));

    cell_type* cuda_cell;
    cuda_malloc(&cuda_cell, M + 1);

    for (uint i = 0; i < N; i++)
    {
        kernel_decorator(
                copy_chunk<cell_type>,
                block_count, threads_per_block, true,
                cells, cuda_cell, i*(M + 1), (M + 1), thread_count
        );
        cuda_memcpy_from_gpu(cell, cuda_cell, M + 1);

        if (max < cell[M])
        {
            max = cell[M];
        }
    }
    free(cell);
    cuda_free(cuda_cell);

    return max;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
long SDM_CS2_S2<cell_type, index_type, summation_type, value_type>::get_activations_num()
{
    long activations = 0;

    cell_type* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));

    cell_type* cuda_cell;
    cuda_malloc(&cuda_cell, M + 1);

    for (uint i = 0; i < N; i++)
    {
        kernel_decorator(
                copy_chunk<cell_type>,
                block_count, threads_per_block, true,
                cells, cuda_cell, i*(M + 1), (M + 1), thread_count
        );
        cuda_memcpy_from_gpu(cell, cuda_cell, M + 1);

        if (cell[M] != 0)
        {
            activations += 1;
        }
        if (cell[M] < 0)
        {
            throw std::invalid_argument("negative activations");
        }
    }
    free(cell);
    cuda_free(cuda_cell);

    return activations;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
void SDM_CS2_S2<cell_type, index_type, summation_type, value_type>::print_state()
{
    cell_type* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));
    index_type* indices_mask = (index_type*)malloc(K * sizeof(index_type));
    bool* bits_mask = (bool*)malloc(K * sizeof(bool));

    cell_type* cuda_cell;
    cuda_malloc(&cuda_cell, M + 1);

    index_type* cuda_indices_mask;
    cuda_malloc(&cuda_indices_mask, K);

    bool* cuda_bits_mask;
    cuda_malloc(&cuda_bits_mask, K);

    for (uint i = 0; i < 25; i++)
    {
        kernel_decorator(
                copy_chunk<index_type>,
                block_count, threads_per_block, true,
                indices, cuda_indices_mask, i*K, K, thread_count
        );
        cuda_memcpy_from_gpu(indices_mask, cuda_indices_mask, K);

        kernel_decorator(
                copy_chunk<bool>,
                block_count, threads_per_block, true,
                bits, cuda_bits_mask, i*K, K, thread_count
        );
        cuda_memcpy_from_gpu(bits_mask, cuda_bits_mask, K);

        std::cout << "[";
        for (uint k = 0; k < K; k++)
        {
            std::cout << "(" << indices_mask[k] << ":" << bits_mask[k] << ")";
        }
        std::cout << "]";

        std::cout << "	---->	";

        kernel_decorator(
                copy_chunk<cell_type>,
                block_count, threads_per_block, true,
                cells, cuda_cell, i*(M + 1), (M + 1), thread_count
        );
        cuda_memcpy_from_gpu(cell, cuda_cell, M + 1);
        for (uint j = 0; j < M; j++)
        {
            std::cout << cell[j] << "  ";
        }
        std::cout << "acts=" << cell[M];
        std::cout << std::endl;
    }
    cuda_free(cuda_cell);
    cuda_free(cuda_indices_mask);
    cuda_free(cuda_bits_mask);

    free(cell);
    free(indices_mask);
    free(bits_mask);
}

#endif // !sdm_cs2_s1_cuh
