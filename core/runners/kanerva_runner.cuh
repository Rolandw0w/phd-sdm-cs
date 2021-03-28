#ifndef kanerva_runner_cuh
#define kanerva_runner_cuh

#include "base_runner.cuh"
#include "../data_ops/data_writer.hpp"
#include "../sdm/sdm_kanerva.cuh"

namespace Runners
{
    class KanervaRunnerParameters : public BaseRunnerParameters
    {
    public:
        uint image_count;
        uint block_count;
        uint threads_per_block;
        uint max_dist;
        uint cells_count;
        uint address_length;
        uint value_length;

        KanervaRunnerParameters(uint image_count, uint block_count, uint threads_per_block,
                                uint max_dist, uint cells_count, uint address_length, uint value_length) :
                image_count(image_count),
                block_count(block_count),
                threads_per_block(threads_per_block),
                max_dist(max_dist),
                cells_count(cells_count),
                address_length(address_length),
                value_length(value_length)
        {}
    };

    class KanervaRunner : public BaseRunner
    {
    public:
        report_map naive(const double confidence, const bool save_images = false, const std::string images_path = "");
        report_map multiple_write(const uint error_bits, const uint write_count);
        report_map iterative_read(const uint iterations_count, const uint error_bits);
        report_map noisy_address(const uint error_bits);
        report_map noisy_address_noisy_value(const uint error_bits);

        void set_data(bool*** d) { this->data = *d; }

        void set_parameters(KanervaRunnerParameters* params) { this->parameters = params; }
    private:
        bool** data;
        KanervaRunnerParameters* parameters;
    };
}


#endif // !kanerva_runner_cuh
