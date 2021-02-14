#ifndef labels_runner_cuh
#define labels_runner_cuh

#include "base_runner.cuh"
#include "data_writer.h"
#include "sdm_labels.cuh"

namespace Runners
{
	class LabelsRunnerParameters : public BaseRunnerParameters
	{
	public:
		uint image_count;
		uint block_count;
		uint threads_per_block;
		uint mask_length;
		uint cells_count;
		uint address_length;
		uint value_length;
		uint labels_count;
		ReadingType reading_type;
		double bio_threshold;

		LabelsRunnerParameters(uint image_count, uint block_count, uint threads_per_block, uint mask_length,
			uint cells_count, uint address_length, uint value_length, uint labels_count,
			ReadingType reading_type, double bio_threshold = 0.0) :
			
			image_count(image_count),
			block_count(block_count),
			threads_per_block(threads_per_block),
			mask_length(mask_length),
			cells_count(cells_count),
			address_length(address_length),
			value_length(value_length),
			labels_count(labels_count),
			reading_type(reading_type),
			bio_threshold(bio_threshold)
		{}
	};

	class LabelsRunner : public BaseRunner
	{
	public:
		report_map naive(const double confidence, const bool save_images = false, const std::string images_path = "");

		void set_data(bool*** data) { this->data = *data; }

		void set_parameters(LabelsRunnerParameters* parameters) { this->parameters = parameters; }
		LabelsRunnerParameters* get_parameters() { return this->parameters; }

		//~CIFAR10Runner();
	private:
		bool** data;
		LabelsRunnerParameters* parameters;
	};
}


#endif // !labels_runner_cuh
