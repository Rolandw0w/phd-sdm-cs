#ifndef sdm_labels_cuh
#define sdm_labels_cuh

#include "sdm_base.cuh"

enum class ReadingType {
	STATISTICAL = 0,
	BIOLOGICAL = 1
};


template<typename cell_type, typename index_type, typename summation_type>
struct SDM_LABELS : SDM_BASE<cell_type, index_type, summation_type>
{
public:
	ulong K;
    ulong L;
    ulong M;
    ulong N;

	cell_type* cells;
	index_type* indices;
	bool* bits;

    ulong block_count;
    ulong threads_per_block;
    ulong thread_count;

	ReadingType reading_type;
	double bio_threshold;

	SDM_LABELS(ulong K, ulong L, ulong M, ulong N, ulong block_count, ulong threads_per_block, ReadingType reading_type, double bio_threshold = 0.0);
	~SDM_LABELS();

	bool* read(const bool* value);
	bool* read(const bool* value, int iter_num);
	bool* read(const bool* value, const bool* address);

	void read_stat(bool* cuda_value, summation_type* cuda_sum, int* cuda_activation_indices, int activated_cells_number, double* cuda_thresholds);
	void read_bio(bool* cuda_value, int* cuda_activation_indices, int activated_cells_number, double* cuda_thresholds);

	void write(const bool* value);
	void write(const bool* value, int times);
	void write(const bool* value, const bool* address);
	void write(const bool* value, const bool* address, int times);

	cell_type get_min_activations();
	cell_type get_max_activations();

	long get_activations_num();
	long get_non_readable();
	long get_non_writable();

	void print_state();
private:
	long non_readable = 0;
	long non_writable = 0;
};

template<typename cell_type, typename index_type, typename summation_type>
SDM_LABELS<cell_type, index_type, summation_type>::SDM_LABELS(ulong K, ulong L, ulong M, ulong N,
	ulong block_count, ulong threads_per_block, ReadingType reading_type, double bio_threshold)
{
	this->K = K;
	this->L = L;
	this->M = M;
	this->N = N;
	this->block_count = block_count;
	this->threads_per_block = threads_per_block;
	this->bio_threshold = bio_threshold;

	thread_count = this->block_count * this->threads_per_block;
	this->reading_type = reading_type;

	cudaMalloc((void**)&cells, N * (M + 1) * sizeof(cell_type));
	cudaMalloc((void**)&indices, K * N * sizeof(index_type));
	//cudaMalloc((void**)&bits, K * N * sizeof(bool));

	init_labels <<<block_count, threads_per_block>>> (cells, indices, bits, K, L, M, N, thread_count);
}

template<typename cell_type, typename index_type, typename summation_type>
SDM_LABELS<cell_type, index_type, summation_type>::~SDM_LABELS()
{
	cudaFree(cells);
	cudaFree(indices);
	//cudaFree(bits);
	cudaDeviceSynchronize();
}

template<typename cell_type, typename index_type, typename summation_type>
bool* SDM_LABELS<cell_type, index_type, summation_type>::read(const bool* value, const bool* address)
{
	bool* cuda_value;
	cudaMalloc((void**)&cuda_value, M * sizeof(bool));
	cudaMemcpy(cuda_value, value, M * sizeof(bool), cudaMemcpyHostToDevice);

	int* cuda_activation_indices;
	cudaMalloc((void**)&cuda_activation_indices, N * sizeof(int));

	summation_type* cuda_sum;
	cudaMalloc((void**)&cuda_sum, M * sizeof(summation_type));
	cudaMemset(cuda_sum, 0, M * sizeof(summation_type));

	bool* cuda_address;
	cudaMalloc((void**)&cuda_address, L * sizeof(bool));
	cudaMemcpy(cuda_address, address, L * sizeof(bool), cudaMemcpyHostToDevice);

	int* cuda_activation_counter;
	cudaMalloc((void**)&cuda_activation_counter, sizeof(int));

	int* activation_counter = (int*)malloc(sizeof(int));
	activation_counter[0] = 0;
	cudaMemcpy(cuda_activation_counter, activation_counter, sizeof(int), cudaMemcpyHostToDevice);

	get_activated_cells_bool<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
		(indices, true, K, M, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter);

	cudaMemcpy(activation_counter, cuda_activation_counter, sizeof(int), cudaMemcpyDeviceToHost);
	int activated_cells_number = activation_counter[0];

	read_jaeckel<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
		(cells, cuda_activation_indices, M, thread_count, cuda_sum, activated_cells_number);

	get_result<summation_type> <<<block_count, threads_per_block>>>
		(cuda_sum, cuda_value, M, thread_count);

	cudaFree(cuda_activation_counter);
	cudaMemset(cuda_sum, 0, M * sizeof(summation_type));

    bool* result = (bool*)malloc(M * sizeof(bool));
	cudaMemcpy(result, cuda_value, M * sizeof(bool), cudaMemcpyDeviceToHost);

	free(activation_counter);

	cudaFree(cuda_address);
	cudaFree(cuda_activation_indices);
	cudaFree(cuda_sum);
	cudaFree(cuda_value);
	cudaDeviceSynchronize();

	return result;
}

template<typename cell_type, typename index_type, typename summation_type>
bool* SDM_LABELS<cell_type, index_type, summation_type>::read(const bool* value)
{
	return this->read(value, 1);
}

template<typename cell_type, typename index_type, typename summation_type>
bool* SDM_LABELS<cell_type, index_type, summation_type>::read(const bool* value, const int iter_num)
{
	bool* cuda_value;
	cudaMalloc((void**)&cuda_value, M * sizeof(bool));
	cudaMemcpy(cuda_value, value, M * sizeof(bool), cudaMemcpyHostToDevice);

	int* cuda_activation_indices;
	cudaMalloc((void**)&cuda_activation_indices, N * sizeof(int));

	summation_type* cuda_sum;
	cudaMalloc((void**)&cuda_sum, M * sizeof(summation_type));
	
	for (int i = 0; i < iter_num; i++)
	{
		int* cuda_activation_counter;
		cudaMalloc((void**)&cuda_activation_counter, sizeof(int));

		int* activation_counter = (int*)malloc(sizeof(int));
		activation_counter[0] = 0;
		cudaMemcpy(cuda_activation_counter, activation_counter, sizeof(int), cudaMemcpyHostToDevice);

		get_activated_cells_bool<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
			(indices, true, K, M, N, thread_count, cuda_value, cuda_activation_indices, cuda_activation_counter);
		cudaDeviceSynchronize();

		cudaMemcpy(activation_counter, cuda_activation_counter, sizeof(int), cudaMemcpyDeviceToHost);
		int activated_cells_number = activation_counter[0];

		if (activated_cells_number == 0)
		{
			non_readable += 1;
			cudaFree(cuda_value);
			cudaFree(cuda_activation_indices);
			cudaFree(cuda_sum);
			cudaFree(cuda_activation_counter);
			
			bool* result = (bool*)malloc(M * sizeof(bool));
			memset(result, 0, M * sizeof(bool));

			return result;
		}
		int* cuda_Ks;
		cudaMalloc((void**)&cuda_Ks, sizeof(int));

		int* Ks = (int*)malloc(sizeof(int));
		Ks[0] = 0;
		cudaMemcpy(cuda_Ks, Ks, sizeof(int), cudaMemcpyHostToDevice);

		sum_array<bool, int> <<<1, threads_per_block>>> (cuda_value, M, cuda_Ks);
		cudaMemcpy(Ks, cuda_Ks, sizeof(int), cudaMemcpyDeviceToHost);

		int K = Ks[0];
		double p1 = (double)K / M;
		double p0 = 1.0 - p1;

		double* cuda_thresholds;
		cudaMalloc((void**)&cuda_thresholds, activated_cells_number * sizeof(double));

		get_thresholds<cell_type> <<<block_count, threads_per_block>>> (cells, cuda_activation_indices,
			activated_cells_number, p0, p1, M, cuda_thresholds, thread_count);

		switch (reading_type) {
		case ReadingType::STATISTICAL:
			read_stat(cuda_value, cuda_sum, cuda_activation_indices, activated_cells_number, cuda_thresholds);
			break;
		case ReadingType::BIOLOGICAL:
			read_bio(cuda_value, cuda_activation_indices, activated_cells_number, cuda_thresholds);
			break;
		}
		
		cudaDeviceSynchronize();

		cudaFree(cuda_activation_counter);
		cudaFree(cuda_Ks);
		cudaFree(cuda_thresholds);
		cudaMemset(cuda_sum, 0, M * sizeof(summation_type));

		free(activation_counter);
		free(Ks);
		
		cudaDeviceSynchronize();
	}

	bool* result = (bool*)malloc(M * sizeof(bool));
	cudaMemcpy(result, cuda_value, M * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaFree(cuda_activation_indices);
	cudaFree(cuda_sum);
	cudaFree(cuda_value);
	cudaDeviceSynchronize();

	return result;
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_LABELS<cell_type, index_type, summation_type>::read_stat(bool* cuda_value, summation_type* cuda_sum,
	int* cuda_activation_indices, int activated_cells_number, double* cuda_thresholds)
{
	auto* thresholds = (double*)malloc(activated_cells_number * sizeof(double));
	cudaMemcpy(thresholds, cuda_thresholds, activated_cells_number * sizeof(double), cudaMemcpyDeviceToHost);

	double threshold = median(thresholds, activated_cells_number);

	read_jaeckel<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
		(cells, cuda_activation_indices, M, thread_count, cuda_sum, activated_cells_number);
	cudaDeviceSynchronize();

	get_result<summation_type> <<<block_count, threads_per_block>>>
		(cuda_sum, cuda_value, M, thread_count, threshold = threshold);

	free(thresholds);
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_LABELS<cell_type, index_type, summation_type>::read_bio(bool* cuda_value,
	int* cuda_activation_indices, int activated_cells_number, double* cuda_thresholds)
{
	bool* cuda_decisions;
	cudaMalloc((void**)&cuda_decisions, activated_cells_number * M * sizeof(bool));

	get_bio_decisions<cell_type> <<<block_count, threads_per_block>>> (cells,
		cuda_decisions, cuda_thresholds, cuda_activation_indices, activated_cells_number, M, thread_count);
	cudaDeviceSynchronize();

	int min_ones_count = ceil(bio_threshold * activated_cells_number);

	get_bio_result<int> <<<block_count, threads_per_block>>> (cuda_decisions,
		cuda_value, min_ones_count, M, activated_cells_number, thread_count);

	cudaFree(cuda_decisions);
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_LABELS<cell_type, index_type, summation_type>::write(const bool* value)
{
	this->write(value, 1);
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_LABELS<cell_type, index_type, summation_type>::write(const bool* value, const int times)
{
	this->write(value, value, times);
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_LABELS<cell_type, index_type, summation_type>::write(const bool* value, const bool* address)
{
	this->write(value, address, 1);
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_LABELS<cell_type, index_type, summation_type>::write(const bool* value, const bool* address, const int times)
{

	bool* cuda_value;
	cudaMalloc((void**)&cuda_value, M * sizeof(bool));
	cudaMemcpy(cuda_value, value, M * sizeof(bool), cudaMemcpyHostToDevice);

	bool* cuda_address;
	cudaMalloc((void**)&cuda_address, L * sizeof(bool));
	cudaMemcpy(cuda_address, address, L * sizeof(bool), cudaMemcpyHostToDevice);

	int* cuda_activation_indices;
	cudaMalloc((void**)&cuda_activation_indices, N * sizeof(int));

	int* cuda_activation_counter;
	cudaMalloc((void**)&cuda_activation_counter, sizeof(int));

	int* activation_counter = (int*)malloc(sizeof(int));
	activation_counter[0] = 0;
	cudaMemcpy(cuda_activation_counter, activation_counter, sizeof(int), cudaMemcpyHostToDevice);

	get_activated_cells_bool<cell_type, index_type, summation_type> <<<block_count, threads_per_block>>>
		(indices, true, K, M, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter);
	cudaDeviceSynchronize();

	cudaMemcpy(activation_counter, cuda_activation_counter, sizeof(int), cudaMemcpyDeviceToHost);
	int activated_cells_number = activation_counter[0];

	if (activated_cells_number == 0)
	{
//		non_writable += 1;
//		for (int i = 0; i < 651; i++)
//			std::cout << value[i];
	}
	else
	{
		cell_type* cuda_to_add;
		cudaMalloc((void**)&cuda_to_add, (M + 1) * sizeof(cell_type));
		get_array_to_add <<<block_count, threads_per_block>>> (cuda_value, cuda_to_add, M, times, thread_count);
		cudaDeviceSynchronize();

		write_jaeckel<cell_type, index_type> <<<block_count, threads_per_block>>>
			(cells, M, thread_count, cuda_to_add, cuda_activation_indices, activated_cells_number);
		cudaFree(cuda_to_add);
	}

	cudaFree(cuda_activation_counter);
	cudaFree(cuda_activation_indices);
	cudaFree(cuda_address);
	cudaFree(cuda_value);
	cudaDeviceSynchronize();

	free(activation_counter);
}

template<typename cell_type, typename index_type, typename summation_type>
cell_type SDM_LABELS<cell_type, index_type, summation_type>::get_min_activations()
{
	cell_type min = 0;

	auto* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));

	cell_type* cuda_cell;
	cudaMalloc((void**)&cuda_cell, (M + 1) * sizeof(cell_type));

	for (ulong i = 0; i < N; i++)
	{
		copy_chunk<cell_type> <<<block_count, threads_per_block>>> (cells, cuda_cell, i * (M + 1), (M + 1), thread_count);
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
cell_type SDM_LABELS<cell_type, index_type, summation_type>::get_max_activations()
{
	cell_type max = 0;

	auto* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));

	cell_type* cuda_cell;
	cudaMalloc((void**)&cuda_cell, (M + 1) * sizeof(cell_type));

	for (ulong i = 0; i < N; i++)
	{
		copy_chunk<cell_type> <<<block_count, threads_per_block>>> (cells, cuda_cell, i * (M + 1), (M + 1), thread_count);
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
long SDM_LABELS<cell_type, index_type, summation_type>::get_activations_num()
{
	long activations = 0;

	auto* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));

	cell_type* cuda_cell;
	cudaMalloc((void**)&cuda_cell, (M + 1) * sizeof(cell_type));

	for (ulong i = 0; i < N; i++)
	{
		copy_chunk<cell_type> <<<block_count, threads_per_block >>> (cells, cuda_cell, i * (M + 1), (M + 1), thread_count);
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
void SDM_LABELS<cell_type, index_type, summation_type>::print_state()
{
	auto* cell = (cell_type*)malloc((M + 1) * sizeof(cell_type));
	auto* indices_mask = (index_type*)malloc(K * sizeof(index_type));
	bool* bits_mask = (bool*)malloc(K * sizeof(bool));

	cell_type* cuda_cell;
	cudaMalloc((void**)&cuda_cell, (M + 1) * sizeof(cell_type));

	index_type* cuda_indices_mask;
	cudaMalloc((void**)&cuda_indices_mask, K * sizeof(index_type));

	bool* cuda_bits_mask;
	cudaMalloc((void**)&cuda_bits_mask, K * sizeof(bool));

	for (ulong i = 0; i < 10; i++)
	{
		copy_chunk<index_type> <<<block_count, threads_per_block>>> (indices, cuda_indices_mask, i * K, K, thread_count);
		cudaMemcpy(indices_mask, cuda_indices_mask, K * sizeof(index_type), cudaMemcpyDeviceToHost);

		copy_chunk<bool> <<<block_count, threads_per_block>>> (bits, cuda_bits_mask, i * K, K, thread_count);
		cudaMemcpy(bits_mask, cuda_bits_mask, K * sizeof(bool), cudaMemcpyDeviceToHost);

		std::cout << "[";
		for (uint k = 0; k < K; k++)
		{
			std::cout << "(" << indices_mask[k] << ":" << bits_mask[k] << ")";
		}
		std::cout << "]";

		std::cout << "	---->	";

		copy_chunk<cell_type> <<<block_count, threads_per_block>>> (cells, cuda_cell, i * (M + 1), (M + 1), thread_count);
		cudaMemcpy(cell, cuda_cell, (M + 1) * sizeof(cell_type), cudaMemcpyDeviceToHost);
		for (uint j = 0; j < 64; j++)
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

template<typename cell_type, typename index_type, typename summation_type>
long SDM_LABELS<cell_type, index_type, summation_type>::get_non_readable()
{
	return non_readable;
}

template<typename cell_type, typename index_type, typename summation_type>
long SDM_LABELS<cell_type, index_type, summation_type>::get_non_writable()
{
	return non_writable;
}
#endif // !sdm_labels_cuh
