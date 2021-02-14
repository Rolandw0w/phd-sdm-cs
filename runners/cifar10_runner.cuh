#ifndef cifar10_runner_cuh
#define cifar10_runner_cuh

#include "base_runner.cuh"
#include "../data_ops/data_writer.h"
#include "../sdm/sdm_jaeckel.cuh"

namespace Runners
{
	class CIFAR10RunnerParameters : public BaseRunnerParameters
	{
	public:
		uint image_count;
		uint block_count;
		uint threads_per_block;
		uint mask_length;
		uint cells_count;
		uint address_length;
		uint value_length;

		CIFAR10RunnerParameters(uint image_count, uint block_count, uint threads_per_block,
			uint mask_length, uint cells_count, uint address_length, uint value_length) :
			image_count(image_count),
			block_count(block_count),
			threads_per_block(threads_per_block),
			mask_length(mask_length),
			cells_count(cells_count),
			address_length(address_length),
			value_length(value_length)
		{}
	};

	class CIFAR10Runner : public BaseRunner
	{
	public:
		report_map naive(const double confidence, const bool save_images = false, const std::string images_path = "");
		report_map multiple_write(const uint error_bits, const uint write_count);
		report_map iterative_read(const uint iterations_count, const uint error_bits);
		report_map noisy_address(const uint error_bits);
		report_map noisy_address_noisy_value(const uint error_bits);

		void set_data(bool*** data) { this->data = *data; }

		void set_parameters(CIFAR10RunnerParameters* parameters) { this->parameters = parameters; }
		CIFAR10RunnerParameters* get_parameters() { return this->parameters; }

		//~CIFAR10Runner();
	private:
		bool** data;
		CIFAR10RunnerParameters* parameters;
	};
}


#endif // !cifar10_runner_cuh
