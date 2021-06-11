#ifndef kanerva_runner_cu
#define kanerva_runner_cu

#include <fstream>

#include "kanerva_runner.cuh"

namespace Runners
{
    report_map KanervaRunner::naive(const std::string& data_path, const std::string& output_path)
    {
        auto start = std::chrono::system_clock::now();
        std::time_t start_time = std::chrono::system_clock::to_time_t(start);
        std::cout << "Started computation at " << std::ctime(&start_time) << std::endl;

        report_map report;

        report.insert({"max_dist", parameters->max_dist});
        report.insert({"value_length", parameters->value_length});
        report.insert({"address_length", parameters->address_length});
        report.insert({"cells_count", parameters->cells_count});
        report.insert({"image_count", parameters->image_count});
        report.insert({"block_count", parameters->block_count});
        report.insert({"threads_per_block", parameters->threads_per_block});
        report.insert({"p0", parameters->p0});

        SDM_KANERVA<short, short, int> sdm(parameters->max_dist, parameters->address_length,
                                             parameters->value_length, parameters->cells_count, parameters->block_count,
                                             parameters->threads_per_block, parameters->p0);

        bool* addr = (bool*) malloc(parameters->cells_count*parameters->address_length*sizeof(bool));
        cuda_memcpy_from_gpu(addr, sdm.addresses, parameters->cells_count*parameters->address_length);

        std::ofstream address_file;
        address_file.open(output_path +
                          "/kanerva_addresses_p0_" + std::to_string(parameters->p0).substr(0, 5) + ".csv");

        for (int i = 0; i < parameters->cells_count; i++)
        {
            for (int j = 0; j < parameters->address_length; j++)
            {
                bool addr_val = addr[i*parameters->address_length + j];
                address_file << addr_val;
            }
            address_file << std::endl;
        }
        address_file.close();
        free(addr);

        uint iters = parameters->image_count / 500;
        for (int i = 0; i < iters; i++)
        {
            long write_time_start = get_current_time_millis();
            uint activations = 0;
            uint left = i*500;
            uint right = left + 500;
            std::cout << "Started writing " << left << " to " << right << std::endl;

            for (uint j = left; j < right; j++)
            {
                uint activated_cells_count = sdm.write(data[j]);
                activations += (activated_cells_count != 0);
            }
            long write_time = get_current_time_millis() - write_time_start;
            double avg_write_time = (double)write_time / 500;
            std::cout << "avg_write_time=" << avg_write_time << ", activations=" << activations << std::endl;
            report.insert({"activations_" + std::to_string(right), activations});
            report.insert({"avg_write_time_" + std::to_string(right), avg_write_time});

            std::cout << "Started reading " << 0 << " to " << right << std::endl;

            std::ofstream restored;
            restored.open(output_path +
                          "/kanerva_D_" + std::to_string(parameters->max_dist) +
                          "_I_" + std::to_string(right) +
                          "_p0_" + std::to_string(parameters->p0).substr(0, 5) + ".csv");
            long read_time_start = get_current_time_millis();
            double sum_dist = 0.0;
            for (uint j = 0; j < right; j++)
            {
                bool* remembered = sdm.read(data[j]);
                int dist = hamming_distance(data[j], remembered, parameters->address_length);
                sum_dist += dist;

                for (uint l = 0; l < parameters->address_length; l++)
                {
                    bool rem = remembered[l];
                    auto sep = (l == parameters->value_length - 1) ? "\n" : ",";
                    restored << rem << sep;
                }

                free(remembered);
            }
            restored.close();
            long read_time = get_current_time_millis() - read_time_start;
            double avg_read_time = (double)read_time / right;
            double avg_dist = sum_dist/right;
            std::cout << "avg_read_time=" << (double)read_time / right << ", avg_dist=" << avg_dist << std::endl;
            report.insert({"avg_read_time_" + std::to_string(right), avg_read_time});
            report.insert({"avg_dist_" + std::to_string(right), avg_dist});
        }

        return report;
    }
}
#endif // !kanerva_runner_cu
