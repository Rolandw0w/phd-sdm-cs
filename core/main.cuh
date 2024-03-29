#ifndef main_cuh
#define main_cuh

#include <iostream>
#include <random>
#include <vector>

#include "data_ops/data_reader.hpp"
#include "data_ops/data_writer.hpp"
#include "utils/utils.hpp"

#include "sdm/sdm_jaeckel.cuh"
#include "sdm/sdm_labels.cuh"

#include "runners/cifar10_runner.cuh"
#include "runners/cs1_runner.cuh"
#include "runners/cs2_runner.cuh"
#include "runners/cs2_s2_runner.cuh"
#include "runners/kanerva_runner.cuh"
#include "runners/labels_runner.cuh"
#include "runners/synth_runner.cuh"

Runners::CIFAR10RunnerParameters* get_cifar10_parameters();
Runners::CS1RunnerParameters* get_cs1_parameters();
Runners::CS2RunnerParameters* get_cs2_parameters();
Runners::CS2S2RunnerParameters* get_cs2_s2_parameters();
Runners::LabelsRunnerParameters* get_labels_parameters(ReadingType reading_type, double bio_threshold = 0.0);
Runners::SynthRunnerParameters* get_synth_parameters();
struct TestData
{
	uint image_num;
	uint block_count;
	uint threads_per_block;
	uint mask_length;
	uint cells_count;

	TestData(uint image_num, uint block_count, uint threads_per_block, uint mask_length, uint cells_count) :
		image_num(image_num),
		block_count(block_count),
		threads_per_block(threads_per_block),
		mask_length(mask_length),
		cells_count(cells_count)
	{}
};


int main(int argc, char** argv);

#endif // !main_cuh
