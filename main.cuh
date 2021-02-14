#ifndef main_cuh
#define main_cuh

#include <iostream>
#include <random>
#include <vector>

#include "data_reader.h"
#include "data_writer.h"
#include "utils.h"

#include "sdm_jaeckel.cuh"
#include "sdm_labels.cuh"

#include "cifar10_runner.cuh"
#include "labels_runner.cuh"

Runners::CIFAR10RunnerParameters* get_cifar10_parameters();
Runners::LabelsRunnerParameters* get_labels_parameters();
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


int main();

#endif // !main_cuh
