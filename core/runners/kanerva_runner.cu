#ifndef kanerva_runner_cu
#define kanerva_runner_cu

#include "kanerva_runner.cuh"

namespace Runners
{
    report_map KanervaRunner::naive(const double confidence, const bool save_images, const std::string images_path)
    {
        report_map report;
        const int required_exact_bits = (int) (parameters->value_length * confidence);

        report.insert({"max_dist", parameters->max_dist});
        report.insert({"value_length", parameters->value_length});
        report.insert({"address_length", parameters->address_length});
        report.insert({"cells_count", parameters->cells_count});
        report.insert({"image_count", parameters->image_count});
        report.insert({"block_count", parameters->block_count});
        report.insert({"threads_per_block", parameters->threads_per_block});
        report.insert({"confidence", confidence});

        SDM_KANERVA<short, short, short> sdm(parameters->max_dist, parameters->address_length,
                                             parameters->value_length, parameters->cells_count, parameters->block_count, parameters->threads_per_block);

        long write_time_start = clock();

        for (uint i = 0; i < parameters->image_count; i++)
        {
            sdm.write(data[i]);
        }
        std::cout << std::endl;

        long write_time = clock() - write_time_start;

        report.insert({"avg_write_time", (double)write_time / parameters->image_count});

        double sum = 0.0;
        int exact_num = 0;

        long read_time_start = clock();

        double weighted_num = 0.0;
        double overall_weight = 0.0;

        int max_weight_index = -1;
        double max_weight = 0;

        ulong all_bits = (ulong) parameters->value_length * parameters->image_count;
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
                uint pow = 7 - (j % 8);
                overall_weight += (1 << pow);
                if (data[i][j] != remembered[j])
                {
                    incorrect_bits += 1;
                    weighted_num += (1 << pow);
                    image_weight += (1 << pow);
                }
            }
            uint exact_bits = parameters->address_length - dist;
            exact_num += (exact_bits >= required_exact_bits) ? 1 : 0;
            sum += dist;

            if (save_images)
            {
                std::string image_number = std::to_string(i+1);

                std::string input_path = images_path;
                input_path.append(image_number);
                input_path.append("_input.bmp");
                save_image_bmp(_strdup(input_path.c_str()), 32, 32, data[i]);

                std::string output_path = images_path;
                output_path.append("_output.bmp");
                save_image_bmp(_strdup(output_path.c_str()), 32, 32, remembered);
            }

            free(remembered);

            if (image_weight > max_weight)
            {
                max_weight = image_weight;
                max_weight_index = i;
            }

        }
        long read_time = clock() - read_time_start;

        report.insert({"avg_read_time", (double)read_time / parameters->image_count});
        report.insert({"avg_dist", sum / parameters->image_count});
        report.insert({"exact_num_paper", exact_num});
        report.insert({"paper_percent", (100.0 * exact_num / parameters->image_count)});
        report.insert({"weighted_num", weighted_num});
        report.insert({"overall_weight", overall_weight});
        report.insert({"weighted_percent", (100 - 100 * weighted_num / overall_weight)});
        report.insert({"avg_weight", weighted_num / parameters->image_count});
        report.insert({"max_weight", max_weight});
        report.insert({"max_weight_index", max_weight_index + 1});
        report.insert({"incorrect_bits", incorrect_bits});
        report.insert({"correct_bits", all_bits - incorrect_bits});
        report.insert({"all_bits", all_bits});
        report.insert({"correct_100", correct100});
        report.insert({"correct_bits_percent", (100 * (double)(all_bits - incorrect_bits) / all_bits)});
//        report.insert({"min_activations", sdm.get_min_activations()});
//        report.insert({"max_activations", sdm.get_max_activations()});
//        report.insert({"activated_cells_count", sdm.get_activations_num()});

        sdm.~SDM_KANERVA();

        return report;
    }

    report_map KanervaRunner::multiple_write(const uint error_bits, const uint write_count)
    {
        report_map report;

        report.insert({"max_dist", parameters->max_dist});
        report.insert({"value_length", parameters->value_length});
        report.insert({"address_length", parameters->address_length});
        report.insert({"cells_count", parameters->cells_count});
        report.insert({"image_count", parameters->image_count});
        report.insert({"block_count", parameters->block_count});
        report.insert({"threads_per_block", parameters->threads_per_block});
        report.insert({"error_bits", error_bits});
        report.insert({"write_count", write_count});

        SDM_KANERVA<short, short, short> sdm(parameters->max_dist, parameters->address_length,
                                             parameters->value_length, parameters->cells_count, parameters->block_count, parameters->threads_per_block);

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
        report.insert({"avg_write_time", (double)write_time / parameters->image_count});

        double sum = 0.0;

        long read_time_start = clock();

        double weighted_num = 0.0;
        double overall_weight = 0.0;

        int max_weight_index = -1;
        double max_weight = 0;

        ulong all_bits = (ulong) parameters->value_length * parameters->image_count;
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
                uint pow = 7 - (j % 8);
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

        report.insert({"avg_read_time", (double)read_time / parameters->image_count});
        report.insert({"avg_dist", sum / parameters->image_count});
        report.insert({"weighted_num", weighted_num});
        report.insert({"overall_weight", overall_weight});
        report.insert({"weighted_percent", (100 - 100 * weighted_num / overall_weight)});
        report.insert({"avg_weight", weighted_num / parameters->image_count});
        report.insert({"max_weight", max_weight});
        report.insert({"max_weight_index", max_weight_index + 1});
        report.insert({"incorrect_bits", incorrect_bits});
        report.insert({"correct_bits", all_bits - incorrect_bits});
        report.insert({"all_bits", all_bits});
        report.insert({"correct_100", correct100});
        report.insert({"correct_bits_percent", (100 * (double)(all_bits - incorrect_bits) / all_bits)});
//        report.insert({"min_activations", sdm.get_min_activations()});
//        report.insert({"max_activations", sdm.get_max_activations()});
//        report.insert({"activated_cells_count", sdm.get_activations_num()});

        sdm.~SDM_KANERVA();

        return report;
    }

    report_map KanervaRunner::iterative_read(const uint iterations_count, const uint error_bits)
    {
        report_map report;

        report.insert({"max_dist", parameters->max_dist});
        report.insert({"value_length", parameters->value_length});
        report.insert({"address_length", parameters->address_length});
        report.insert({"cells_count", parameters->cells_count});
        report.insert({"image_count", parameters->image_count});
        report.insert({"block_count", parameters->block_count});
        report.insert({"threads_per_block", parameters->threads_per_block});
        report.insert({"iterations_count", iterations_count});
        report.insert({"error_bits", error_bits});

        bool** noisy_data = (bool**)malloc(parameters->image_count * sizeof(bool*));
        for (uint i = 0; i < parameters->image_count; i++)
        {
            noisy_data[i] = noise(data[i], parameters->value_length, error_bits);
        }

        SDM_KANERVA<short, short, short> sdm(parameters->max_dist, parameters->address_length,
                                             parameters->value_length, parameters->cells_count, parameters->block_count, parameters->threads_per_block);

        long write_time_start = clock();

        for (uint i = 0; i < parameters->image_count; i++)
        {
            sdm.write(data[i]);
        }

        long write_time = clock() - write_time_start;
        report.insert({"avg_write_time", (double)write_time / parameters->image_count});

        double sum = 0.0;

        long read_time_start = clock();

        double weighted_num = 0.0;
        double overall_weight = 0.0;

        int max_weight_index = -1;
        double max_weight = 0;

        ulong all_bits = (ulong) parameters->value_length * parameters->image_count;
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
                uint pow = 7 - (j % 8);
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

        report.insert({"avg_read_time", (double)read_time / parameters->image_count});
        report.insert({"avg_dist", sum / parameters->image_count});
        report.insert({"weighted_num", weighted_num});
        report.insert({"overall_weight", overall_weight});
        report.insert({"weighted_percent", (100 - 100 * weighted_num / overall_weight)});
        report.insert({"avg_weight", weighted_num / parameters->image_count});
        report.insert({"max_weight", max_weight});
        report.insert({"max_weight_index", max_weight_index + 1});
        report.insert({"incorrect_bits", incorrect_bits});
        report.insert({"correct_bits", all_bits - incorrect_bits});
        report.insert({"all_bits", all_bits});
        report.insert({"correct_100", correct100});
        report.insert({"correct_bits_percent", (100 * (double)(all_bits - incorrect_bits) / all_bits)});
//        report.insert({"min_activations", sdm.get_min_activations()});
//        report.insert({"max_activations", sdm.get_max_activations()});
//        report.insert({"activated_cells_count", sdm.get_activations_num()});

        sdm.~SDM_KANERVA();
        free(noisy_data);

        return report;
    }

    report_map KanervaRunner::noisy_address(const uint error_bits)
    {
        report_map report;

        report.insert({"max_dist", parameters->max_dist});
        report.insert({"value_length", parameters->value_length});
        report.insert({"address_length", parameters->address_length});
        report.insert({"cells_count", parameters->cells_count});
        report.insert({"image_count", parameters->image_count});
        report.insert({"block_count", parameters->block_count});
        report.insert({"threads_per_block", parameters->threads_per_block});
        report.insert({"error_bits", error_bits});

        SDM_KANERVA<short, short, short> sdm(parameters->max_dist, parameters->address_length,
                                             parameters->value_length, parameters->cells_count, parameters->block_count, parameters->threads_per_block);

        long write_time_start = clock();

        for (uint i = 0; i < parameters->image_count; i++)
        {
            bool* noisy_address = noise(data[i], parameters->address_length, error_bits);

            sdm.write(data[i], noisy_address);

            free(noisy_address);
        }

        long write_time = clock() - write_time_start;
        report.insert({"avg_write_time", (double)write_time / parameters->image_count});

        double sum = 0.0;

        long read_time_start = clock();

        double weighted_num = 0.0;
        double overall_weight = 0.0;

        int max_weight_index = -1;
        double max_weight = 0;

        ulong all_bits = (ulong) parameters->value_length * parameters->image_count;
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
                uint pow = 7 - (j % 8);
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

        report.insert({"avg_read_time", (double)read_time / parameters->image_count});
        report.insert({"avg_dist", sum / parameters->image_count});
        report.insert({"weighted_num", weighted_num});
        report.insert({"overall_weight", overall_weight});
        report.insert({"weighted_percent", (100 - 100 * weighted_num / overall_weight)});
        report.insert({"avg_weight", weighted_num / parameters->image_count});
        report.insert({"max_weight", max_weight});
        report.insert({"max_weight_index", max_weight_index + 1});
        report.insert({"incorrect_bits", incorrect_bits});
        report.insert({"correct_bits", all_bits - incorrect_bits});
        report.insert({"all_bits", all_bits});
        report.insert({"correct_100", correct100});
        report.insert({"correct_bits_percent", (100 * (double)(all_bits - incorrect_bits) / all_bits)});
//        report.insert({"min_activations", sdm.get_min_activations()});
//        report.insert({"max_activations", sdm.get_max_activations()});
//        report.insert({"activated_cells_count", sdm.get_activations_num()});

        sdm.~SDM_KANERVA();

        return report;
    }

    report_map KanervaRunner::noisy_address_noisy_value(const uint error_bits)
    {
        report_map report;

        report.insert({"max_dist", parameters->max_dist});
        report.insert({"value_length", parameters->value_length});
        report.insert({"address_length", parameters->address_length});
        report.insert({"cells_count", parameters->cells_count});
        report.insert({"image_count", parameters->image_count});
        report.insert({"block_count", parameters->block_count});
        report.insert({"threads_per_block", parameters->threads_per_block});
        report.insert({"error_bits", error_bits});

        SDM_KANERVA<short, short, short> sdm(parameters->max_dist, parameters->address_length,
                                             parameters->value_length, parameters->cells_count, parameters->block_count, parameters->threads_per_block);

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
        report.insert({"avg_write_time", (double)write_time / parameters->image_count});

        double sum = 0.0;

        long read_time_start = clock();

        double weighted_num = 0.0;
        double overall_weight = 0.0;

        int max_weight_index = -1;
        double max_weight = 0;

        ulong all_bits = (ulong) parameters->value_length * parameters->image_count;
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
                uint pow = 7 - (j % 8);
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

        report.insert({"avg_read_time", (double)read_time / parameters->image_count});
        report.insert({"avg_dist", sum / parameters->image_count});
        report.insert({"weighted_num", weighted_num});
        report.insert({"overall_weight", overall_weight});
        report.insert({"weighted_percent", (100 - 100 * weighted_num / overall_weight)});
        report.insert({"avg_weight", weighted_num / parameters->image_count});
        report.insert({"max_weight", max_weight});
        report.insert({"max_weight_index", max_weight_index + 1});
        report.insert({"incorrect_bits", incorrect_bits});
        report.insert({"correct_bits", all_bits - incorrect_bits});
        report.insert({"all_bits", all_bits});
        report.insert({"correct_100", correct100});
        report.insert({"correct_bits_percent", (100 * (double)(all_bits - incorrect_bits) / all_bits)});
//        report.insert({"min_activations", sdm.get_min_activations()});
//        report.insert({"max_activations", sdm.get_max_activations()});
//        report.insert({"activated_cells_count", sdm.get_activations_num()});

        sdm.~SDM_KANERVA();

        return report;
    }
}
#endif // !kanerva_runner_cu
