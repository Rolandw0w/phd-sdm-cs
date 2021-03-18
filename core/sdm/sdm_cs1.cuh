#ifndef sdm_cs1_cuh
#define sdm_cs1_cuh

#include "sdm_base.cuh"


template<typename cell_type, typename index_type, typename summation_type, typename value_type>
struct SDM_CS1// : SDM_BASE<cell_type, index_type, summation_type>
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

    SDM_CS1(uint K, uint L, uint M, uint N, uint block_count, uint threads_per_block);
    ~SDM_CS1();

    double* read(const value_type* value);
    double* read(const value_type* value, int iter_num);
    double* read2(const value_type* value);
    double* read2(const value_type* value, int iter_num);

    int write(const value_type* value);
    int write(const value_type* value, int times);
    int write(const value_type* value, const value_type* address);
    int write(const value_type* value, const value_type* address, int times);

    cell_type get_min_activations();
    cell_type get_max_activations();

    long get_activations_num();

    void print_state();
};

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
SDM_CS1<cell_type, index_type, summation_type, value_type>::SDM_CS1(uint K, uint L, uint M, uint N, uint block_count, uint threads_per_block)
{
    this->K = K;
    this->L = L;
    this->M = M;
    this->N = N;
    this->block_count = block_count;
    this->threads_per_block = threads_per_block;

    thread_count = this->block_count * this->threads_per_block;

    cudaMalloc((void**)&cells, N * (M + 1) * sizeof(cell_type));
    cudaMalloc((void**)&indices, K * N * sizeof(index_type));
    cudaMalloc((void**)&bits, K * N * sizeof(bool));
    check_errors<int>("init_labels");

    init_jaeckel<<<block_count, threads_per_block>>> (cells, indices, bits, K, L, M, N, thread_count);
    check_errors<int>("init_labels");
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
SDM_CS1<cell_type, index_type, summation_type, value_type>::~SDM_CS1()
{
    cudaFree(cells);
    cudaFree(indices);
    cudaFree(bits);
    cudaDeviceSynchronize();
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
double* SDM_CS1<cell_type, index_type, summation_type, value_type>::read(const value_type* value)
{
    return this->read(value, 1);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
double* SDM_CS1<cell_type, index_type, summation_type, value_type>::read(const value_type* value, const int iter_num)
{
    value_type* cuda_value;
    cudaMalloc((void **)&cuda_value, M * sizeof(value_type));
    check_errors<int>();
    cudaMemcpy(cuda_value, value, M * sizeof(value_type), cudaMemcpyHostToDevice);
    check_errors<int>();

    int* cuda_activation_indices;
    cudaMalloc((void **)&cuda_activation_indices, N * sizeof(int));
    check_errors<int>();

    double* cuda_sum;
    cudaMalloc((void **)&cuda_sum, M * sizeof(double));
    check_errors<int>();
    cudaMemset(cuda_sum, 0, M * sizeof(double));
    check_errors<int>();

    for (int i = 0; i < iter_num; i++)
    {
        int* cuda_activation_counter;
        cudaMalloc((void **)&cuda_activation_counter, sizeof(int));
        check_errors<int>();

        int* activation_counter = (int*)malloc(sizeof(int));
        activation_counter[0] = 0;
        cudaMemcpy(cuda_activation_counter, activation_counter, sizeof(int), cudaMemcpyHostToDevice);
        check_errors<int>();

        get_activated_cells_cs1<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
                (indices, bits, K, M, N, thread_count, cuda_value, cuda_activation_indices, cuda_activation_counter);
        cudaDeviceSynchronize();
        check_errors<int>();

        cudaMemcpy(activation_counter, cuda_activation_counter, sizeof(int), cudaMemcpyDeviceToHost);
        check_errors<int>();
        int activated_cells_number = activation_counter[0];
        //std::cout << "activated_cells_number=" << activated_cells_number << " ";
        //std::cout << "activated_cells_number_read=" << activated_cells_number << std::endl;
        //std::cout << activated_cells_number << ",";
//        std::cout << std::endl;
//        std::cout << std::endl;
//        std::cout << std::endl;
//        std::cout << activated_cells_number << ",";
        //std::cout << activated_cells_number << ",";
//		if (activated_cells_number != 0)
//        {
//		    for(int j = 0; j < M; j++)
//            {
//		        std::cout << value[j];
//            }
//		    std::cout << std::endl;
//        }
        int* cuda_sum_act;
        cudaMalloc((void **)&cuda_sum_act, sizeof(int));
        get_acts_sum<cell_type, int><<<block_count, threads_per_block>>>
                (cells, cuda_activation_indices, M, cuda_activation_counter, cuda_sum_act);
        cudaDeviceSynchronize();
        check_errors<int>();

        int* sum_act = (int*) malloc(sizeof(int));
        cudaMemcpy(sum_act, cuda_sum_act, sizeof(int), cudaMemcpyDeviceToHost);
        //std::cout << "sumact=" << sum_act[0] << std::endl;

        read_cs1<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
                (cells, cuda_activation_indices, M, thread_count, cuda_sum, cuda_activation_counter, cuda_sum_act);
        cudaDeviceSynchronize();
        check_errors<int>();

        cudaFree(cuda_activation_counter);
        //cudaMemset(cuda_sum, 0, M * sizeof(summation_type));

        free(activation_counter);

        cudaDeviceSynchronize();
        check_errors<int>();
    }
    cudaDeviceSynchronize();
    check_errors<int>();
    double* result = (double*)malloc(M * sizeof(double));
    cudaMemcpy(result, cuda_sum, M * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    check_errors<int>();

    cudaFree(cuda_activation_indices);
    cudaFree(cuda_sum);
    cudaFree(cuda_value);
    cudaDeviceSynchronize();
    check_errors<int>();

//    std::cout << std::endl;
//    for (int i = 0; i < M; i++)
//        std::cout << result[i] << ",";
//    std::cout << std::endl;
//    for (int i = 0; i < M; i++)
//        std::cout << value[i] << ",";
//    std::cout << std::endl;

    return result;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
double* SDM_CS1<cell_type, index_type, summation_type, value_type>::read2(const value_type* value)
{
    return this->read2(value, 1);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
double* SDM_CS1<cell_type, index_type, summation_type, value_type>::read2(const value_type* value, const int iter_num)
{
    value_type* cuda_value;
    cudaMalloc((void **)&cuda_value, M * sizeof(value_type));
    check_errors<int>();
    cudaMemcpy(cuda_value, value, M * sizeof(value_type), cudaMemcpyHostToDevice);
    check_errors<int>();

    int* cuda_activation_indices;
    cudaMalloc((void **)&cuda_activation_indices, N * sizeof(int));
    check_errors<int>();

    double* cuda_sum;
    cudaMalloc((void **)&cuda_sum, M * sizeof(double));
    check_errors<int>();
    cudaMemset(cuda_sum, 0, M * sizeof(double));
    check_errors<int>();

    for (int i = 0; i < iter_num; i++)
    {
        int* cuda_activation_counter;
        cudaMalloc((void **)&cuda_activation_counter, sizeof(int));
        check_errors<int>();

        int* activation_counter = (int*)malloc(sizeof(int));
        activation_counter[0] = 0;
        cudaMemcpy(cuda_activation_counter, activation_counter, sizeof(int), cudaMemcpyHostToDevice);
        check_errors<int>();

        get_activated_cells_cs1<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
                (indices, bits, K, M, N, thread_count, cuda_value, cuda_activation_indices, cuda_activation_counter);
        cudaDeviceSynchronize();
        check_errors<int>();

        cudaMemcpy(activation_counter, cuda_activation_counter, sizeof(int), cudaMemcpyDeviceToHost);
        check_errors<int>();
        int activated_cells_number = activation_counter[0];
        //std::cout << activated_cells_number << ",";
//        std::cout << std::endl;
//        std::cout << std::endl;
//        std::cout << std::endl;
//        std::cout << activated_cells_number << ",";
        //std::cout << activated_cells_number << ",";
//		if (activated_cells_number != 0)
//        {
//		    for(int j = 0; j < M; j++)
//            {
//		        std::cout << value[j];
//            }
//		    std::cout << std::endl;
//        }

        read_cs1_2<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
                (cells, cuda_activation_indices, M, thread_count, cuda_sum, activated_cells_number);
        cudaDeviceSynchronize();
        check_errors<int>();

        cudaFree(cuda_activation_counter);
        //cudaMemset(cuda_sum, 0, M * sizeof(summation_type));

        free(activation_counter);

        cudaDeviceSynchronize();
        check_errors<int>();
    }
    cudaDeviceSynchronize();
    check_errors<int>();
    double* result = (double*)malloc(M * sizeof(double));
    cudaMemcpy(result, cuda_sum, M * sizeof(double), cudaMemcpyDeviceToHost);
    check_errors<int>();

    cudaFree(cuda_activation_indices);
    cudaFree(cuda_sum);
    cudaFree(cuda_value);
    cudaDeviceSynchronize();
    check_errors<int>();

//    std::cout << std::endl;
//    for (int i = 0; i < M; i++)
//        std::cout << result[i] << ",";
//    std::cout << std::endl;
//    for (int i = 0; i < M; i++)
//        std::cout << value[i] << ",";
//    std::cout << std::endl;

    return result;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
int SDM_CS1<cell_type, index_type, summation_type, value_type>::write(const value_type *value)
{
    return this->write(value, 1);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
int SDM_CS1<cell_type, index_type, summation_type, value_type>::write(const value_type *value, const int times)
{
    return this->write(value, value, times);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
int SDM_CS1<cell_type, index_type, summation_type, value_type>::write(const value_type *value, const value_type *address)
{
    return this->write(value, address, 1);
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
int SDM_CS1<cell_type, index_type, summation_type, value_type>::write(const value_type *value, const value_type *address, const int times)
{

    value_type* cuda_value;
    cudaMalloc((void **)&cuda_value, M * sizeof(value_type));
    check_errors<int>();
    cudaMemcpy(cuda_value, value, M * sizeof(value_type), cudaMemcpyHostToDevice);
    check_errors<int>();

    value_type* cuda_address;
    cudaMalloc((void **)&cuda_address, L * sizeof(value_type));
    check_errors<int>();
    cudaMemcpy(cuda_address, address, L * sizeof(value_type), cudaMemcpyHostToDevice);
    check_errors<int>();

//    cell_type* cuda_to_add;
//    cudaMalloc((void **)&cuda_to_add, (M + 1) * sizeof(cell_type));
//    check_errors<int>();
//    get_array_to_add <<<block_count, threads_per_block>>> (cuda_value, cuda_to_add, M, times, thread_count);
//    check_errors<int>();
//    cudaDeviceSynchronize();

    int* cuda_activation_indices;
    cudaMalloc((void **)&cuda_activation_indices, N * sizeof(int));
    check_errors<int>();

    int* cuda_activation_counter;
    cudaMalloc((void **)&cuda_activation_counter, sizeof(int));
    check_errors<int>();

    int* activation_counter = (int*)malloc(sizeof(int));
    activation_counter[0] = 0;
    cudaMemcpy(cuda_activation_counter, activation_counter, sizeof(int), cudaMemcpyHostToDevice);
    check_errors<int>();

    get_activated_cells_cs1<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
            (indices, bits, K, M, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter);
    check_errors<int>();
    cudaDeviceSynchronize();

    cudaMemcpy(activation_counter, cuda_activation_counter, sizeof(int), cudaMemcpyDeviceToHost);
    check_errors<int>();
    int activated_cells_number = activation_counter[0];
    //std::cout << "activated_cells_number_write=" << activated_cells_number << std::endl;

    write_cs1<cell_type, index_type, value_type> <<<block_count, threads_per_block>>>
            (cells, M, thread_count, cuda_value, cuda_activation_indices, activated_cells_number);
    cudaDeviceSynchronize();
    check_errors<int>();

    cudaFree(cuda_activation_counter);
    cudaFree(cuda_activation_indices);
    cudaFree(cuda_address);
    //cudaFree(cuda_to_add);
    cudaFree(cuda_value);
    cudaDeviceSynchronize();
    check_errors<int>();

    free(activation_counter);

    return activated_cells_number;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
cell_type SDM_CS1<cell_type, index_type, summation_type, value_type>::get_min_activations()
{
    cell_type min = 0;

    cell_type* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));

    cell_type* cuda_cell;
    cudaMalloc((void **)&cuda_cell, (M + 1) * sizeof(cell_type));

    for (uint i = 0; i < N; i++)
    {
        copy_chunk<cell_type> <<<block_count, threads_per_block>>> (cells, cuda_cell, i*(M + 1), (M + 1), thread_count);
        cudaMemcpy(cell, cuda_cell, (M + 1) * sizeof(cell_type), cudaMemcpyDeviceToHost);

        if (min > cell[M])
        {
            min = cell[M];
        }
    }
    free(cell);
    cudaFree(cuda_cell);

    return min;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
cell_type SDM_CS1<cell_type, index_type, summation_type, value_type>::get_max_activations()
{
    cell_type max = 0;

    cell_type* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));

    cell_type* cuda_cell;
    cudaMalloc((void **)&cuda_cell, (M + 1) * sizeof(cell_type));

    for (uint i = 0; i < N; i++)
    {
        copy_chunk<cell_type> <<<block_count, threads_per_block>>> (cells, cuda_cell, i*(M + 1), (M + 1), thread_count);
        cudaMemcpy(cell, cuda_cell, (M + 1) * sizeof(cell_type), cudaMemcpyDeviceToHost);

        if (max < cell[M])
        {
            max = cell[M];
        }
    }
    free(cell);
    cudaFree(cuda_cell);

    return max;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
long SDM_CS1<cell_type, index_type, summation_type, value_type>::get_activations_num()
{
    long activations = 0;

    cell_type* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));

    cell_type* cuda_cell;
    cudaMalloc((void **)&cuda_cell, (M + 1) * sizeof(cell_type));

    for (uint i = 0; i < N; i++)
    {
        copy_chunk<cell_type> <<<block_count, threads_per_block>>> (cells, cuda_cell, i*(M + 1), (M + 1), thread_count);
        cudaMemcpy(cell, cuda_cell, (M + 1) * sizeof(cell_type), cudaMemcpyDeviceToHost);

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
    cudaFree(cuda_cell);

    return activations;
}

template<typename cell_type, typename index_type, typename summation_type, typename value_type>
void SDM_CS1<cell_type, index_type, summation_type, value_type>::print_state()
{
    cell_type* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));
    index_type* indices_mask = (index_type*)malloc(K * sizeof(index_type));
    bool* bits_mask = (bool*)malloc(K * sizeof(bool));

    cell_type* cuda_cell;
    cudaMalloc((void **)&cuda_cell, (M + 1) * sizeof(cell_type));

    index_type* cuda_indices_mask;
    cudaMalloc((void **)&cuda_indices_mask, K * sizeof(index_type));

    bool* cuda_bits_mask;
    cudaMalloc((void **)&cuda_bits_mask, K * sizeof(bool));

    for (uint i = 0; i < 25; i++)
    {
        copy_chunk<index_type> <<<block_count, threads_per_block>>> (indices, cuda_indices_mask, i*K, K, thread_count);
        cudaMemcpy(indices_mask, cuda_indices_mask, K * sizeof(index_type), cudaMemcpyDeviceToHost);

        copy_chunk<bool> <<<block_count, threads_per_block>>> (bits, cuda_bits_mask, i*K, K, thread_count);
        cudaMemcpy(bits_mask, cuda_bits_mask, K * sizeof(bool), cudaMemcpyDeviceToHost);

        std::cout << "[";
        for (uint k = 0; k < K; k++)
        {
            std::cout << "(" << indices_mask[k] << ":" << bits_mask[k] << ")";
        }
        std::cout << "]";

        std::cout << "	---->	";

        copy_chunk<cell_type> <<<block_count, threads_per_block>>> (cells, cuda_cell, i*(M + 1), (M + 1), thread_count);
        cudaMemcpy(cell, cuda_cell, (M + 1) * sizeof(cell_type), cudaMemcpyDeviceToHost);
        for (uint j = 0; j < M; j++)
        {
            std::cout << cell[j] << "  ";
        }
        std::cout << "acts=" << cell[M];
        std::cout << std::endl;
    }
    cudaFree(cuda_cell);
    cudaFree(cuda_indices_mask);
    cudaFree(cuda_bits_mask);

    free(cell);
    free(indices_mask);
    free(bits_mask);
}

#endif // !sdm_cs1_cuh
