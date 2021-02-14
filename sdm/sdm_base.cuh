#ifndef sdm_base_cuh
#define sdm_base_cuh

#include "../cuda/kernels.cuh"


template<typename cell_type, typename index_type, typename summation_type>
struct SDM_BASE
{
public:
	cell_type* cells;
	index_type* indices;
	bool* bits;

	uint block_count;
	uint threads_per_block;
	uint thread_count;

	bool* read(const bool* value);

	void write(const bool* value);
};
#endif // smd_base_cuh