#ifndef cs1_runner_cuh
#define cs1_runner_cuh

#include "base_runner.cuh"
#include "../sdm/sdm_cs1.cuh"
#include "../utils/utils.hpp"

namespace Runners
{
    class CS1RunnerParameters : public BaseRunnerParameters
    {
    public:
        uint image_count;
        uint images_read;
        uint block_count;
        uint threads_per_block;
        uint mask_length;
        uint cells_count;
        uint address_length;
        uint value_length;
        uint labels_count;
        uint target_count;
        uint bits_per_num;

        CS1RunnerParameters(uint image_count, uint images_read, uint block_count, uint threads_per_block, uint mask_length,
                            uint cells_count, uint address_length, uint value_length, uint labels_count,
                            uint target_count, uint bits_per_num) :

                image_count(image_count),
                images_read(images_read),
                block_count(block_count),
                threads_per_block(threads_per_block),
                mask_length(mask_length),
                cells_count(cells_count),
                address_length(address_length),
                value_length(value_length),
                labels_count(labels_count),
                target_count(target_count),
                bits_per_num(bits_per_num)
        {}
    };

    class CS1Runner : public BaseRunner
    {
    public:
        report_map naive(const std::string& data_path, const std::string& output_path);

        void set_data(bool** d) { this->data = *d; }

        void set_parameters(CS1RunnerParameters* params) { this->parameters = params; }
        CS1RunnerParameters* get_parameters() { return this->parameters; }
    private:
        bool* data;
        CS1RunnerParameters* parameters;
    };
}


#endif //cs1_runner_cuh
