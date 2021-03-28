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

	cuda_malloc(&cells, N * (M + 1));
    cuda_malloc(&indices, K * N);

    kernel_decorator(
            init_labels<cell_type, index_type>,
            block_count, threads_per_block, true,
            cells, indices, K, L, M, N, thread_count
    );
}

template<typename cell_type, typename index_type, typename summation_type>
SDM_LABELS<cell_type, index_type, summation_type>::~SDM_LABELS()
{
	cuda_free(cells);
	cuda_free(indices);
}

template<typename cell_type, typename index_type, typename summation_type>
bool* SDM_LABELS<cell_type, index_type, summation_type>::read(const bool* value, const bool* address)
{
	bool* cuda_value;
	cuda_malloc(&cuda_value, M);
	cuda_memcpy_to_gpu(cuda_value, value, M);

	int* cuda_activation_indices;
	cuda_malloc(&cuda_activation_indices, N);

	summation_type* cuda_sum;
	cuda_malloc(&cuda_sum, M);
	cuda_memset(cuda_sum, 0, M);

	bool* cuda_address;
	cuda_malloc(&cuda_address, L);
	cuda_memcpy_to_gpu(cuda_address, address, L);

	int* cuda_activation_counter;
	cuda_malloc(&cuda_activation_counter, 1);

	int* activation_counter = (int*)malloc(sizeof(int));
	activation_counter[0] = 0;
	cuda_memcpy_to_gpu(cuda_activation_counter, activation_counter, 1);

	kernel_decorator(
            get_activated_cells_bool<cell_type, index_type, summation_type>,
            block_count, threads_per_block, true,
            indices, true, K, M, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter
	);

	cuda_memcpy_from_gpu(activation_counter, cuda_activation_counter, 1);
	int activated_cells_number = activation_counter[0];

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
            cuda_sum, cuda_value, M, thread_count
    );

	cuda_memset(cuda_sum, 0, M);

    bool* result = (bool*)malloc(M * sizeof(bool));
	cuda_memcpy_from_gpu(result, cuda_value, M);

	free(activation_counter);

    cuda_free(cuda_activation_counter);
    cuda_free(cuda_address);
    cuda_free(cuda_activation_indices);
    cuda_free(cuda_sum);
    cuda_free(cuda_value);

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
	cuda_malloc(&cuda_value, M);
	cuda_memcpy_to_gpu(cuda_value, value, M);

	int* cuda_activation_indices;
    cuda_malloc(&cuda_activation_indices, N);

	summation_type* cuda_sum;
    cuda_malloc(&cuda_sum, M);
	
	for (int i = 0; i < iter_num; i++)
	{
		int* cuda_activation_counter;
        cuda_malloc(&cuda_activation_counter, 1);

		int* activation_counter = (int*)malloc(sizeof(int));
		activation_counter[0] = 0;
        cuda_memcpy_to_gpu(cuda_activation_counter, activation_counter, 1);

        kernel_decorator(
                get_activated_cells_bool<cell_type, index_type, summation_type>,
                block_count, threads_per_block, true,
                indices, true, K, M, N, thread_count, cuda_value, cuda_activation_indices, cuda_activation_counter
        );

		cuda_memcpy_from_gpu(activation_counter, cuda_activation_counter, 1);
		int activated_cells_number = activation_counter[0];

		if (activated_cells_number == 0)
		{
            bool* result = (bool*)malloc(M * sizeof(bool));
            memset(result, 0, M * sizeof(bool));
			non_readable += 1;

			free(activation_counter);
			cuda_free(cuda_value);
            cuda_free(cuda_activation_indices);
            cuda_free(cuda_sum);
            cuda_free(cuda_activation_counter);

			return result;
		}
		int* cuda_Ks;
		cuda_malloc(&cuda_Ks, 1);

		int* Ks = (int*)malloc(sizeof(int));
		Ks[0] = 0;
		cuda_memcpy_to_gpu(cuda_Ks, Ks, 1);

		sum_array<bool, int> <<<1, threads_per_block>>> (cuda_value, M, cuda_Ks);
		cuda_memcpy_from_gpu(Ks, cuda_Ks, 1);

		int K_ = Ks[0];
		double p1 = (double)K_ / M;
		double p0 = 1.0 - p1;

		double* cuda_thresholds;
		cuda_malloc(&cuda_thresholds, activated_cells_number);

		kernel_decorator(
                get_thresholds<cell_type>,
                block_count, threads_per_block, true,
                cells, cuda_activation_indices, activated_cells_number, p0, p1, M, cuda_thresholds, thread_count
		);

		switch (reading_type) {
		case ReadingType::STATISTICAL:
			read_stat(cuda_value, cuda_sum, cuda_activation_indices, activated_cells_number, cuda_thresholds);
			break;
		case ReadingType::BIOLOGICAL:
			read_bio(cuda_value, cuda_activation_indices, activated_cells_number, cuda_thresholds);
			break;
		}

        free(activation_counter);
        free(Ks);

        cuda_free(cuda_activation_counter);
        cuda_free(cuda_Ks);
        cuda_free(cuda_thresholds);
		cuda_memset(cuda_sum, 0, M);
	}

	bool* result = (bool*)malloc(M * sizeof(bool));
	cuda_memcpy_from_gpu(result, cuda_value, M);

    cuda_free(cuda_activation_indices);
    cuda_free(cuda_sum);
    cuda_free(cuda_value);

	return result;
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_LABELS<cell_type, index_type, summation_type>::read_stat(bool* cuda_value, summation_type* cuda_sum,
	int* cuda_activation_indices, int activated_cells_number, double* cuda_thresholds)
{
	auto* thresholds = (double*)malloc(activated_cells_number * sizeof(double));
	cuda_memcpy_from_gpu(thresholds, cuda_thresholds, activated_cells_number);

	double threshold = median(thresholds, activated_cells_number);

    kernel_decorator(
            read_jaeckel<cell_type, index_type, summation_type>,
            block_count, threads_per_block, true,
            cells, cuda_activation_indices, M, thread_count, cuda_sum, activated_cells_number
    );

    kernel_decorator(
            get_result<summation_type>,
            block_count, threads_per_block, true,
            cuda_sum, cuda_value, M, thread_count, threshold
    );

	free(thresholds);
}

template<typename cell_type, typename index_type, typename summation_type>
void SDM_LABELS<cell_type, index_type, summation_type>::read_bio(bool* cuda_value,
	int* cuda_activation_indices, int activated_cells_number, double* cuda_thresholds)
{
	bool* cuda_decisions;
	cuda_malloc(&cuda_decisions, activated_cells_number * M);

    kernel_decorator(
            get_bio_decisions<cell_type>,
            block_count, threads_per_block, true,
            cells, cuda_decisions, cuda_thresholds, cuda_activation_indices, activated_cells_number, M, thread_count
    );

	int min_ones_count = ceil(bio_threshold * activated_cells_number);

    kernel_decorator(
            get_bio_result<int>,
            block_count, threads_per_block, true,
            cuda_decisions, cuda_value, min_ones_count, M, activated_cells_number, thread_count
    );

	cuda_free(cuda_decisions);
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
	cuda_malloc(&cuda_value, M);
	cuda_memcpy_to_gpu(cuda_value, value, M);

	bool* cuda_address;
    cuda_malloc(&cuda_address, L);
    cuda_memcpy_to_gpu(cuda_address, address, L);

	int* cuda_activation_indices;
    cuda_malloc(&cuda_activation_indices, N);

	int* cuda_activation_counter;
    cuda_malloc(&cuda_activation_counter, 1);

	int* activation_counter = (int*)malloc(sizeof(int));
	activation_counter[0] = 0;
	cuda_memcpy_to_gpu(cuda_activation_counter, activation_counter, 1);

	kernel_decorator(
	        get_activated_cells_bool<cell_type, index_type, summation_type>,
	        block_count, threads_per_block, true,
	        indices, true, K, M, N, thread_count, cuda_address, cuda_activation_indices, cuda_activation_counter
	);

	cuda_memcpy_from_gpu(activation_counter, cuda_activation_counter, 1);
	int activated_cells_number = activation_counter[0];

	if (activated_cells_number == 0)
	{

	}
	else
	{
		cell_type* cuda_to_add;
        cuda_malloc(&cuda_to_add, (M + 1));

        kernel_decorator(
                get_array_to_add<cell_type>,
                block_count, threads_per_block, true,
                cuda_value, cuda_to_add, M, times, thread_count
        );

        kernel_decorator(
                write_jaeckel<cell_type, index_type>,
                block_count, threads_per_block, true,
                cells, M, thread_count, cuda_to_add, cuda_activation_indices, activated_cells_number
        );

		cuda_free(cuda_to_add);
	}

    free(activation_counter);

    cuda_free(cuda_activation_counter);
    cuda_free(cuda_activation_indices);
    cuda_free(cuda_address);
    cuda_free(cuda_value);
}

template<typename cell_type, typename index_type, typename summation_type>
cell_type SDM_LABELS<cell_type, index_type, summation_type>::get_min_activations()
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

template<typename cell_type, typename index_type, typename summation_type>
cell_type SDM_LABELS<cell_type, index_type, summation_type>::get_max_activations()
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

template<typename cell_type, typename index_type, typename summation_type>
long SDM_LABELS<cell_type, index_type, summation_type>::get_activations_num()
{
    ulong activations = 0;

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

template<typename cell_type, typename index_type, typename summation_type>
void SDM_LABELS<cell_type, index_type, summation_type>::print_state()
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
