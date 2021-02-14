#ifndef cifar10_runner_cu
#define cifar10_runner_cu

#include "cifar10_runner.cuh"

namespace Runners
{
	report_map CIFAR10Runner::naive(const double confidence, const bool save_images, const std::string images_path)
	{
		report_map report;
		CIFAR10RunnerParameters* parameters = this->get_parameters();
		const int required_exact_bits = parameters->value_length * confidence;

		report["mask_length"] = parameters->mask_length;
		report["value_length"] = parameters->value_length;
		report["address_length"] = parameters->address_length;
		report["cells_count"] = parameters->cells_count;
		report["image_count"] = parameters->image_count;
		report["block_count"] = parameters->block_count;
		report["threads_per_block"] = parameters->threads_per_block;
		report["confidence"] = confidence;

		SDM_JAECKEL<short, short, short> sdm(parameters->mask_length, parameters->address_length,
			parameters->value_length, parameters->cells_count, parameters->block_count, parameters->threads_per_block);
		int* permutations = (int*)malloc(parameters->value_length * sizeof(int));

		for (uint i = 0; i < parameters->value_length; i++)
		{
			//int octet_index = i % 8;
			//int octet_num = i / 8;
			//int new_index = 8 * octet_num + (7 - octet_index);
			permutations[i] = i;
		}
		sdm.set_permutations(permutations);

		long write_time_start = clock();

		for (uint i = 0; i < parameters->image_count; i++)
		{
			sdm.write(data[i]);
		}

		long write_time = clock() - write_time_start;

		report["avg_write_time"] = (double)write_time / parameters->image_count;

		double sum = 0.0;
		int exact_num = 0;

		long read_time_start = clock();

		double weighted_num = 0.0;
		double overall_weight = 0.0;

		int max_weight_index = -1;
		double max_weight = 0;

		long all_bits = parameters->value_length * parameters->image_count;
		long incorrect_bits = 0;
		long correct100 = 0;

		for (uint i = 0; i < parameters->image_count; i++)
		{
			double image_weight = 0;
			bool* remembered = sdm.read(data[i]);
			int dist = hamming_distance(data[i], remembered, parameters->address_length);
			if (dist == 0)
				correct100 += 1;

			for (uint j = 0; j < parameters->address_length; j++)
			{
				int pow = 7 - (j % 8);
				overall_weight += (1 << pow);
				if (data[i][j] != remembered[j])
				{
					incorrect_bits += 1;
					weighted_num += (1 << pow);
					image_weight += (1 << pow);
				}
			}
			int exact_bits = parameters->address_length - dist;
			exact_num += (exact_bits >= required_exact_bits) ? 1 : 0;
			sum += dist;

			if (save_images)
			{
				std::string image_number = std::to_string(i+1);

				std::string input_path = images_path;
				input_path.append(image_number);
				input_path.append("_input.bmp");
				save_image_bmp(strdup(input_path.c_str()), 32, 32, data[i]);
				
				std::string output_path = images_path;
				output_path.append("_output.bmp");
				save_image_bmp(strdup(output_path.c_str()), 32, 32, remembered);
			}
			
			free(remembered);

			if (image_weight > max_weight)
			{
				max_weight = image_weight;
				max_weight_index = i;
			}

		}
		long read_time = clock() - read_time_start;

		report["avg_read_time"] = (double)read_time / parameters->image_count;
		report["avg_dist"] = sum / parameters->image_count;
		report["exact_num_paper"] = exact_num;
		report["paper_percent"] = (100.0 * exact_num / parameters->image_count);
		report["weighted_num"] = weighted_num;
		report["overall_weight"] = overall_weight;
		report["weighted_percent"] = (100 - 100 * weighted_num / overall_weight);
		report["avg_weight"] = weighted_num / parameters->image_count;
		report["max_weight"] = max_weight;
		report["max_weight_index"] = max_weight_index + 1;
		report["incorrect_bits"] = incorrect_bits;
		report["correct_bits"] = all_bits - incorrect_bits;
		report["all_bits"] = all_bits;
		report["correct_100"] = correct100;
		report["correct_bits_percent"] = (100 * (double)(all_bits - incorrect_bits) / all_bits);
		report["min_activations"] = sdm.get_min_activations();
		report["max_activations"] = sdm.get_max_activations();
		report["activated_cells_count"] = sdm.get_activations_num();

		sdm.~SDM_JAECKEL();
		free(permutations);

		return report;
	}

	report_map CIFAR10Runner::multiple_write(const uint error_bits, const uint write_count)
	{
		report_map report;
		CIFAR10RunnerParameters* parameters = this->get_parameters();

		report["mask_length"] = parameters->mask_length;
		report["value_length"] = parameters->value_length;
		report["address_length"] = parameters->address_length;
		report["cells_count"] = parameters->cells_count;
		report["image_count"] = parameters->image_count;
		report["block_count"] = parameters->block_count;
		report["threads_per_block"] = parameters->threads_per_block;
		report["error_bits"] = error_bits;
		report["write_count"] = write_count;

		SDM_JAECKEL<short, short, short> sdm(parameters->mask_length, parameters->address_length,
			parameters->value_length, parameters->cells_count, parameters->block_count, parameters->threads_per_block);
		int* permutations = (int*)malloc(parameters->value_length * sizeof(int));

		for (uint i = 0; i < parameters->value_length; i++)
		{
			//int octet_index = i % 8;
			//int octet_num = i / 8;
			//int new_index = 8 * octet_num + (7 - octet_index);
			permutations[i] = i;
		}
		sdm.set_permutations(permutations);

		long write_time_start = clock();

		for (uint i = 0; i < parameters->image_count; i++)
		{
			for (uint j = 0; j < write_count; j++)
			{
				bool* noised_data = noise(data[i], parameters->value_length, error_bits);
				sdm.write(noised_data);
				free(noised_data);
			}
		}

		long write_time = clock() - write_time_start;
		report["avg_write_time"] = (double)write_time / parameters->image_count;

		double sum = 0.0;

		long read_time_start = clock();

		double weighted_num = 0.0;
		double overall_weight = 0.0;

		int max_weight_index = -1;
		double max_weight = 0;

		long all_bits = parameters->value_length * parameters->image_count;
		long incorrect_bits = 0;
		long correct100 = 0;

		for (uint i = 0; i < parameters->image_count; i++)
		{
			double image_weight = 0;
			bool* remembered = sdm.read(data[i]);
			int dist = hamming_distance(data[i], remembered, parameters->address_length);
			if (dist == 0)
				correct100 += 1;

			for (uint j = 0; j < parameters->address_length; j++)
			{
				int pow = 7 - (j % 8);
				overall_weight += (1 << pow);
				if (data[i][j] != remembered[j])
				{
					incorrect_bits += 1;
					weighted_num += (1 << pow);
					image_weight += (1 << pow);
				}
			}
			sum += dist;

			free(remembered);

			if (image_weight > max_weight)
			{
				max_weight = image_weight;
				max_weight_index = i;
			}

		}
		long read_time = clock() - read_time_start;

		report["avg_read_time"] = (double)read_time / parameters->image_count;
		report["avg_dist"] = sum / parameters->image_count;
		report["weighted_num"] = weighted_num;
		report["overall_weight"] = overall_weight;
		report["weighted_percent"] = (100 - 100 * weighted_num / overall_weight);
		report["avg_weight"] = weighted_num / parameters->image_count;
		report["max_weight"] = max_weight;
		report["max_weight_index"] = max_weight_index + 1;
		report["incorrect_bits"] = incorrect_bits;
		report["correct_bits"] = all_bits - incorrect_bits;
		report["all_bits"] = all_bits;
		report["correct_100"] = correct100;
		report["correct_bits_percent"] = (100 * (double)(all_bits - incorrect_bits) / all_bits);
		report["min_activations"] = sdm.get_min_activations();
		report["max_activations"] = sdm.get_max_activations();
		report["activated_cells_count"] = sdm.get_activations_num();

		free(permutations);
		sdm.~SDM_JAECKEL();

		return report;
	}

	report_map CIFAR10Runner::iterative_read(const uint iterations_count, const uint error_bits)
	{
		report_map report;
		CIFAR10RunnerParameters* parameters = this->get_parameters();

		report["mask_length"] = parameters->mask_length;
		report["value_length"] = parameters->value_length;
		report["address_length"] = parameters->address_length;
		report["cells_count"] = parameters->cells_count;
		report["image_count"] = parameters->image_count;
		report["block_count"] = parameters->block_count;
		report["threads_per_block"] = parameters->threads_per_block;
		report["iterations_count"] = iterations_count;
		report["error_bits"] = error_bits;

		bool** noisy_data = (bool**)malloc(parameters->image_count * sizeof(bool*));
		for (uint i = 0; i < parameters->image_count; i++)
		{
			noisy_data[i] = noise(data[i], parameters->value_length, error_bits);
		}

		SDM_JAECKEL<short, short, short> sdm(parameters->mask_length, parameters->address_length,
			parameters->value_length, parameters->cells_count, parameters->block_count, parameters->threads_per_block);
		int* permutations = (int*)malloc(parameters->value_length * sizeof(int));

		for (uint i = 0; i < parameters->value_length; i++)
		{
			//int octet_index = i % 8;
			//int octet_num = i / 8;
			//int new_index = 8 * octet_num + (7 - octet_index);
			permutations[i] = i;
		}
		sdm.set_permutations(permutations);

		long write_time_start = clock();

		for (uint i = 0; i < parameters->image_count; i++)
		{
			sdm.write(data[i]);
		}

		long write_time = clock() - write_time_start;
		report["avg_write_time"] = (double)write_time / parameters->image_count;

		double sum = 0.0;

		long read_time_start = clock();

		double weighted_num = 0.0;
		double overall_weight = 0.0;

		int max_weight_index = -1;
		double max_weight = 0;

		long all_bits = parameters->value_length * parameters->image_count;
		long incorrect_bits = 0;
		long correct100 = 0;

		for (uint i = 0; i < parameters->image_count; i++)
		{
			double image_weight = 0;
			bool* remembered = sdm.read(noisy_data[i], iterations_count);
			int dist = hamming_distance(data[i], remembered, parameters->address_length);
			if (dist == 0)
				correct100 += 1;

			for (uint j = 0; j < parameters->address_length; j++)
			{
				int pow = 7 - (j % 8);
				overall_weight += (1 << pow);
				if (data[i][j] != remembered[j])
				{
					incorrect_bits += 1;
					weighted_num += (1 << pow);
					image_weight += (1 << pow);
				}
			}
			sum += dist;

			free(remembered);

			if (image_weight > max_weight)
			{
				max_weight = image_weight;
				max_weight_index = i;
			}

		}
		long read_time = clock() - read_time_start;

		report["avg_read_time"] = (double)read_time / parameters->image_count;
		report["avg_dist"] = sum / parameters->image_count;
		report["weighted_num"] = weighted_num;
		report["overall_weight"] = overall_weight;
		report["weighted_percent"] = (100 - 100 * weighted_num / overall_weight);
		report["avg_weight"] = weighted_num / parameters->image_count;
		report["max_weight"] = max_weight;
		report["max_weight_index"] = max_weight_index + 1;
		report["incorrect_bits"] = incorrect_bits;
		report["correct_bits"] = all_bits - incorrect_bits;
		report["all_bits"] = all_bits;
		report["correct_100"] = correct100;
		report["correct_bits_percent"] = (100 * (double)(all_bits - incorrect_bits) / all_bits);
		report["min_activations"] = sdm.get_min_activations();
		report["max_activations"] = sdm.get_max_activations();
		report["activated_cells_count"] = sdm.get_activations_num();

		free(permutations);
		sdm.~SDM_JAECKEL();
		free(noisy_data);

		return report;
	}

	report_map CIFAR10Runner::noisy_address(const uint error_bits)
	{
		report_map report;
		CIFAR10RunnerParameters* parameters = this->get_parameters();

		report["mask_length"] = parameters->mask_length;
		report["value_length"] = parameters->value_length;
		report["address_length"] = parameters->address_length;
		report["cells_count"] = parameters->cells_count;
		report["image_count"] = parameters->image_count;
		report["block_count"] = parameters->block_count;
		report["threads_per_block"] = parameters->threads_per_block;
		report["error_bits"] = error_bits;

		SDM_JAECKEL<short, short, short> sdm(parameters->mask_length, parameters->address_length,
			parameters->value_length, parameters->cells_count, parameters->block_count, parameters->threads_per_block);
		int* permutations = (int*)malloc(parameters->value_length * sizeof(int));

		for (uint i = 0; i < parameters->value_length; i++)
		{
			//int octet_index = i % 8;
			//int octet_num = i / 8;
			//int new_index = 8 * octet_num + (7 - octet_index);
			permutations[i] = i;
		}
		sdm.set_permutations(permutations);

		long write_time_start = clock();

		for (uint i = 0; i < parameters->image_count; i++)
		{
			bool* noisy_address = noise(data[i], parameters->address_length, error_bits);

			sdm.write(data[i], noisy_address);

			free(noisy_address);
		}

		long write_time = clock() - write_time_start;
		report["avg_write_time"] = (double)write_time / parameters->image_count;

		double sum = 0.0;

		long read_time_start = clock();

		double weighted_num = 0.0;
		double overall_weight = 0.0;

		int max_weight_index = -1;
		double max_weight = 0;

		long all_bits = parameters->value_length * parameters->image_count;
		long incorrect_bits = 0;
		long correct100 = 0;

		for (uint i = 0; i < parameters->image_count; i++)
		{
			double image_weight = 0;

			bool* noisy_address = noise(data[i], parameters->address_length, error_bits);
			bool* remembered = sdm.read(data[i], noisy_address);
			free(noisy_address);

			int dist = hamming_distance(data[i], remembered, parameters->address_length);
			if (dist == 0)
				correct100 += 1;

			for (uint j = 0; j < parameters->address_length; j++)
			{
				int pow = 7 - (j % 8);
				overall_weight += (1 << pow);
				if (data[i][j] != remembered[j])
				{
					incorrect_bits += 1;
					weighted_num += (1 << pow);
					image_weight += (1 << pow);
				}
			}
			sum += dist;

			free(remembered);

			if (image_weight > max_weight)
			{
				max_weight = image_weight;
				max_weight_index = i;
			}

		}
		long read_time = clock() - read_time_start;

		report["avg_read_time"] = (double)read_time / parameters->image_count;
		report["avg_dist"] = sum / parameters->image_count;
		report["weighted_num"] = weighted_num;
		report["overall_weight"] = overall_weight;
		report["weighted_percent"] = (100 - 100 * weighted_num / overall_weight);
		report["avg_weight"] = weighted_num / parameters->image_count;
		report["max_weight"] = max_weight;
		report["max_weight_index"] = max_weight_index + 1;
		report["incorrect_bits"] = incorrect_bits;
		report["correct_bits"] = all_bits - incorrect_bits;
		report["all_bits"] = all_bits;
		report["correct_100"] = correct100;
		report["correct_bits_percent"] = (100 * (double)(all_bits - incorrect_bits) / all_bits);
		report["min_activations"] = sdm.get_min_activations();
		report["max_activations"] = sdm.get_max_activations();
		report["activated_cells_count"] = sdm.get_activations_num();

		free(permutations);
		sdm.~SDM_JAECKEL();

		return report;
	}

	report_map CIFAR10Runner::noisy_address_noisy_value(const uint error_bits)
	{
		report_map report;
		CIFAR10RunnerParameters* parameters = this->get_parameters();

		report["mask_length"] = parameters->mask_length;
		report["value_length"] = parameters->value_length;
		report["address_length"] = parameters->address_length;
		report["cells_count"] = parameters->cells_count;
		report["image_count"] = parameters->image_count;
		report["block_count"] = parameters->block_count;
		report["threads_per_block"] = parameters->threads_per_block;
		report["error_bits"] = error_bits;

		SDM_JAECKEL<short, short, short> sdm(parameters->mask_length, parameters->address_length,
			parameters->value_length, parameters->cells_count, parameters->block_count, parameters->threads_per_block);
		int* permutations = (int*)malloc(parameters->value_length * sizeof(int));

		for (uint i = 0; i < parameters->value_length; i++)
		{
			//int octet_index = i % 8;
			//int octet_num = i / 8;
			//int new_index = 8 * octet_num + (7 - octet_index);
			permutations[i] = i;
		}
		sdm.set_permutations(permutations);

		long write_time_start = clock();

		for (uint i = 0; i < parameters->image_count; i++)
		{
			bool* noisy_address = noise(data[i], parameters->address_length, error_bits);
			bool* noisy_value = noise(data[i], parameters->address_length, error_bits);

			sdm.write(noisy_value, noisy_address);

			free(noisy_address);
			free(noisy_value);
		}

		long write_time = clock() - write_time_start;
		report["avg_write_time"] = (double)write_time / parameters->image_count;

		double sum = 0.0;

		long read_time_start = clock();

		double weighted_num = 0.0;
		double overall_weight = 0.0;

		int max_weight_index = -1;
		double max_weight = 0;

		long all_bits = parameters->value_length * parameters->image_count;
		long incorrect_bits = 0;
		long correct100 = 0;

		for (uint i = 0; i < parameters->image_count; i++)
		{
			double image_weight = 0;

			bool* noisy_address = noise(data[i], parameters->address_length, error_bits);
			bool* noisy_value = noise(data[i], parameters->value_length, error_bits);

			bool* remembered = sdm.read(noisy_value, noisy_address);

			free(noisy_address);
			free(noisy_value);

			int dist = hamming_distance(data[i], remembered, parameters->address_length);
			if (dist == 0)
				correct100 += 1;

			for (uint j = 0; j < parameters->address_length; j++)
			{
				int pow = 7 - (j % 8);
				overall_weight += (1 << pow);
				if (data[i][j] != remembered[j])
				{
					incorrect_bits += 1;
					weighted_num += (1 << pow);
					image_weight += (1 << pow);
				}
			}
			sum += dist;

			free(remembered);

			if (image_weight > max_weight)
			{
				max_weight = image_weight;
				max_weight_index = i;
			}

		}
		long read_time = clock() - read_time_start;

		report["avg_read_time"] = (double)read_time / parameters->image_count;
		report["avg_dist"] = sum / parameters->image_count;
		report["weighted_num"] = weighted_num;
		report["overall_weight"] = overall_weight;
		report["weighted_percent"] = (100 - 100 * weighted_num / overall_weight);
		report["avg_weight"] = weighted_num / parameters->image_count;
		report["max_weight"] = max_weight;
		report["max_weight_index"] = max_weight_index + 1;
		report["incorrect_bits"] = incorrect_bits;
		report["correct_bits"] = all_bits - incorrect_bits;
		report["all_bits"] = all_bits;
		report["correct_100"] = correct100;
		report["correct_bits_percent"] = (100 * (double)(all_bits - incorrect_bits) / all_bits);
		report["min_activations"] = sdm.get_min_activations();
		report["max_activations"] = sdm.get_max_activations();
		report["activated_cells_count"] = sdm.get_activations_num();

		free(permutations);
		sdm.~SDM_JAECKEL();

		return report;
	}
}
#endif // !cifar10_runner_cu
