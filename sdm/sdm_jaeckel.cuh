#ifndef sdm_jaeckel_cuh
#define sdm_jaeckel_cuh

#include "sdm_base.cuh"


template<typename cell_type, typename index_type, typename summation_type>
struct SDM_JAECKEL : SDM_BASE<cell_type, index_type, summation_type>
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

	SDM_JAECKEL(uint K, uint L, uint M, uint N, uint block_count, uint threads_per_block);
	~SDM_JAECKEL();

	bool* read(const bool* value);
	bool* read(const bool* value, int iter_count);
	bool* read(const bool* value, const bool* address);

	void write(const bool *value);
	void write(const bool *value, const int times);
	void write(const bool *value, const bool *address);
	void write(const bool *value, const bool *address, const int times);

	cell_type get_min_activations();
	cell_type get_max_activations();

	long get_activations_num();

	void print_state();
};

template<typename cell_type, typename index_type, typename summation_type>
SDM_JAECKEL<cell_type, index_type, summation_type>::SDM_JAECKEL(uint K, uint L, uint M, uint N, uint block_count, uint threads_per_block)
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

	init_jaeckel<<<block_count, threads_per_block>>> (cells, indices, bits, K, L, M, N, thread_count);
}

template<typename cell_type, typename index_type, typename summation_type>
SDM_JAECKEL<cell_type, index_type, summation_type>::~SDM_JAECKEL()
{
	cudaFree(cells);
	cudaFree(indices);
	cudaFree(bits);
	cudaDeviceSynchronize();
}

template<typename cell_type, typename index_type, typename summation_type>
bool* SDM_JAECKEL<cell_type, index_type, summation_type>::read(const bool* value, const bool* address)
{
	bool* cuda_value;
	cudaMalloc((void **)&cuda_value, M * sizeof(bool));
	cudaMemcpy(cuda_value, value, M * sizeof(bool), cudaMemcpyHostToDevice);

	int* cuda_activation_indices;
	cudaMalloc((void **)&cuda_activation_indices, N * sizeof(int));

	summation_type* cuda_sum;
	cudaMalloc((void **)&cuda_sum, M * sizeof(summation_type));
	cudaMemset(cuda_sum, 0, M * sizeof(summation_type));

	bool* cuda_result;
	cudaMalloc((void**)&cuda_result, M * sizeof(bool));

	bool* cuda_address;
	cudaMalloc((void **)&cuda_address, L * sizeof(bool));
	cudaMemcpy(cuda_address, address, L * sizeof(bool), cudaMemcpyHostToDevice);

	int* cuda_activation_counter;
	cudaMalloc((void **)&cuda_activation_counter, sizeof(int));

	int* activation_counter = (int*)malloc(sizeof(int));
	activation_counter[0] = 0;
	cudaMemcpy(cuda_activation_counter, activation_counter, sizeof(int), cudaMemcpyHostToDevice);

	get_activated_cells<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
		(indices, bits, K, M, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter);

	cudaMemcpy(activation_counter, cuda_activation_counter, sizeof(int), cudaMemcpyDeviceToHost);
	int activated_cells_number = activation_counter[0];

	read_jaeckel<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
		(cells, cuda_activation_indices, M, thread_count, cuda_sum, activated_cells_number);

	get_result<summation_type> <<<block_count, threads_per_block>>>
		(cuda_sum, cuda_value, M, thread_count);

	cudaFree(cuda_activation_counter);
	cudaMemset(cuda_sum, 0, M * sizeof(summation_type));

	bool* result = (bool*)malloc(M * sizeof(bool));
	cudaMemcpy(result, cuda_result, M * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

	free(activation_counter);

	cudaFree(cuda_address);
	cudaFree(cuda_activation_indices);
	cudaFree(cuda_result);
	cudaFree(cuda_sum);
	cudaFree(cuda_value);
	cudaDeviceSynchronize();

	return result;
}

template<typename cell_type, typename index_type, typename summation_type>
bool* SDM_JAECKEL<cell_type, index_type, summation_type>::read(const bool* value)
{
	return this->read(value, 1);
}

template<typename cell_type, typename index_type, typename summation_type>
bool* SDM_JAECKEL<cell_type, index_type, summation_type>::read(const bool* value, const int iter_num)
{
	bool* cuda_value;
	cudaMalloc((void **)&cuda_value, M * sizeof(bool));
    check_errors<int>();
	cudaMemcpy(cuda_value, value, M * sizeof(bool), cudaMemcpyHostToDevice);
    check_errors<int>();

	int* cuda_activation_indices;
	cudaMalloc((void **)&cuda_activation_indices, N * sizeof(int));
    check_errors<int>();

	summation_type* cuda_sum;
	cudaMalloc((void **)&cuda_sum, M * sizeof(summation_type));
    check_errors<int>();
	cudaMemset(cuda_sum, 0, M * sizeof(summation_type));
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

		get_activated_cells<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
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

		read_jaeckel<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
			(cells, cuda_activation_indices, M, thread_count, cuda_sum, activated_cells_number);
		cudaDeviceSynchronize();
        check_errors<int>();
        summation_type* sum = (summation_type*)malloc(M*sizeof(summation_type));
        cudaMemcpy(sum, cuda_sum, M * sizeof(summation_type), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        check_errors<int>();
//        std::cout << std::endl;
//        std::cout << std::endl;
//        std::cout << std::endl;
//        for (int j = 0; j < 10; j++)
//            std::cout << sum[j] << ",";
		    //std::cout << std::endl;

		get_result<summation_type> <<<block_count, threads_per_block>>>
			(cuda_sum, cuda_value, M, thread_count);
		cudaDeviceSynchronize();
        check_errors<int>();

		cudaFree(cuda_activation_counter);
		cudaMemset(cuda_sum, 0, M * sizeof(summation_type));

		free(activation_counter);

		cudaDeviceSynchronize();
        check_errors<int>();
	}
	cudaDeviceSynchronize();
    check_errors<int>();
	bool* result = (bool*)malloc(M * sizeof(bool));
	cudaMemcpy(result, cuda_value, M * sizeof(bool), cudaMemcpyDeviceToHost);
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

template<typename cell_type, typename index_type, typename summation_type>
void SDM_JAECKEL<cell_type, index_type, summation_type>::write(const bool *value)
{
	this->write(value, 1);
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_JAECKEL<cell_type, index_type, summation_type>::write(const bool *value, const int times)
{
	this->write(value, value, times);
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_JAECKEL<cell_type, index_type, summation_type>::write(const bool *value, const bool *address)
{
	this->write(value, address, 1);
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_JAECKEL<cell_type, index_type, summation_type>::write(const bool *value, const bool *address, const int times)
{

	bool* cuda_value;
	cudaMalloc((void **)&cuda_value, M * sizeof(bool));
    check_errors<int>();
	cudaMemcpy(cuda_value, value, M * sizeof(bool), cudaMemcpyHostToDevice);
    check_errors<int>();

	bool* cuda_address;
	cudaMalloc((void **)&cuda_address, L * sizeof(bool));
    check_errors<int>();
	cudaMemcpy(cuda_address, address, L * sizeof(bool), cudaMemcpyHostToDevice);
    check_errors<int>();

	cell_type* cuda_to_add;
	cudaMalloc((void **)&cuda_to_add, (M + 1) * sizeof(cell_type));
    check_errors<int>();
	get_array_to_add <<<block_count, threads_per_block>>> (cuda_value, cuda_to_add, M, times, thread_count);
    check_errors<int>();
	cudaDeviceSynchronize();

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

	get_activated_cells<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
		(indices, bits, K, M, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter);
    check_errors<int>();
	cudaDeviceSynchronize();

	cudaMemcpy(activation_counter, cuda_activation_counter, sizeof(int), cudaMemcpyDeviceToHost);
    check_errors<int>();
	int activated_cells_number = activation_counter[0];
	//std::cout << activated_cells_number << ",";

	write_jaeckel<cell_type, index_type> <<<block_count, threads_per_block>>>
		(cells, M, thread_count, cuda_to_add, cuda_activation_indices, activated_cells_number);
	cudaDeviceSynchronize();
    check_errors<int>();

	cudaFree(cuda_activation_counter);
	cudaFree(cuda_activation_indices);
	cudaFree(cuda_address);
	cudaFree(cuda_to_add);
	cudaFree(cuda_value);
	cudaDeviceSynchronize();
    check_errors<int>();

	free(activation_counter);
}

template<typename cell_type, typename index_type, typename summation_type>
cell_type SDM_JAECKEL<cell_type, index_type, summation_type>::get_min_activations()
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

template<typename cell_type, typename index_type, typename summation_type>
cell_type SDM_JAECKEL<cell_type, index_type, summation_type>::get_max_activations()
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

template<typename cell_type, typename index_type, typename summation_type>
long SDM_JAECKEL<cell_type, index_type, summation_type>::get_activations_num()
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

template<typename cell_type, typename index_type, typename summation_type>
void SDM_JAECKEL<cell_type, index_type, summation_type>::print_state()
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

	for (uint i = 0; i < 10; i++)
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
		for (uint j = 0; j < M + 1; j++)
		{
			std::cout << cell[j] << "  ";
		}
		std::cout << std::endl;
	}
	cudaFree(cuda_cell);
	cudaFree(cuda_indices_mask);
	cudaFree(cuda_bits_mask);

	free(cell);
	free(indices_mask);
	free(bits_mask);
}

#endif // !sdm_jaeckel_cuh
