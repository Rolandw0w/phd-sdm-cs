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
        report.insert({"images_read", parameters->images_read});
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

        long write_time_start = clock();
        uint activations = 0;

        std::cout << "Started writing ";
        for (uint i = 0; i < parameters->images_read; i++)
        {
            uint activated_cells_count = sdm.write(data[i]);
            activations += (activated_cells_count != 0);
            if ((i+1) % 100 == 0)
                std::cout << (i+1) << " ";
        }
        std::cout << std::endl;
        //sdm.print_state();

        long write_time = clock() - write_time_start;

        report.insert({"avg_write_time", (double)write_time / parameters->images_read});

        double sum = 0.0;

        long read_time_start = clock();

        ulong all_bits = (ulong) parameters->value_length * parameters->images_read;
        long incorrect_bits = 0;
        long correct100 = 0;

        std::ofstream restored;
        restored.open(output_path +
                      "/kanerva_D_" + std::to_string(parameters->max_dist) +
                      "_I_" + std::to_string(parameters->images_read) +
                      "_p0_" + std::to_string(parameters->p0).substr(0, 5) + ".csv");
        std::cout << "Started reading ";
        for (uint i = 0; i < parameters->images_read; i++)
        {
            bool* remembered = sdm.read(data[i]);
            int dist = hamming_distance(data[i], remembered, parameters->address_length);
            if (dist == 0)
                correct100 += 1;

            for (uint j = 0; j < parameters->address_length; j++)
            {
                bool rem = remembered[j];
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                restored << rem << sep;
            }
            sum += dist;

            free(remembered);
            if ((i+1) % 100 == 0)
                std::cout << (i+1) << " ";
        }
        std::cout << std::endl;
        long read_time = clock() - read_time_start;
        restored.close();

        report.insert({"avg_read_time", (double)read_time / parameters->images_read});
        report.insert({"activations", activations});
        report.insert({"avg_dist", sum / parameters->images_read});
        report.insert({"incorrect_bits", incorrect_bits});
        report.insert({"correct_bits", all_bits - incorrect_bits});
        report.insert({"all_bits", all_bits});
        report.insert({"correct_100", correct100});
        report.insert({"correct_bits_percent", (100 * (double)(all_bits - incorrect_bits) / all_bits)});
//        report.insert({"min_activations", sdm.get_min_activations()});
//        report.insert({"max_activations", sdm.get_max_activations()});
//        report.insert({"activated_cells_count", sdm.get_activations_num()});

        // sdm.~SDM_KANERVA();

        return report;
    }
}
#endif // !kanerva_runner_cu
