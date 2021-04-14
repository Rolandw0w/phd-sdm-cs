#ifndef kanerva_runner_cuh
#define kanerva_runner_cuh

#include <chrono>

#include "base_runner.cuh"
#include "../data_ops/data_writer.hpp"
#include "../sdm/sdm_kanerva.cuh"

namespace Runners
{
    class KanervaRunnerParameters : public BaseRunnerParameters
    {
    public:
        uint image_count;
        uint images_read;
        uint block_count;
        uint threads_per_block;
        uint max_dist;
        uint cells_count;
        uint address_length;
        uint value_length;
        double p0;

        KanervaRunnerParameters(uint image_count, uint images_read, uint block_count, uint threads_per_block,
                                uint max_dist, uint cells_count, uint address_length, uint value_length, double p0) :
                image_count(image_count),
                images_read(images_read),
                block_count(block_count),
                threads_per_block(threads_per_block),
                max_dist(max_dist),
                cells_count(cells_count),
                address_length(address_length),
                value_length(value_length),
                p0(p0)
        {}
    };

    class KanervaRunner : public BaseRunner
    {
    public:
        report_map naive(const std::string& data_path, const std::string& output_path);

        void set_data(bool*** d) { this->data = *d; }

        void set_parameters(KanervaRunnerParameters* params) { this->parameters = params; }
    private:
        bool** data;
        KanervaRunnerParameters* parameters;
    };
}


#endif // !kanerva_runner_cuh
