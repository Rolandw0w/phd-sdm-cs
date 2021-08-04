#ifndef synth_runner_cuh
#define synth_runner_cuh

#include "base_runner.cuh"
#include "../data_ops/data_reader.hpp"
#include "../sdm/sdm_cs1.cuh"
#include "../sdm/sdm_cs2.cuh"
#include "../sdm/sdm_jaeckel.cuh"
#include "../sdm/sdm_labels.cuh"
#include "../utils/utils.hpp"

namespace Runners
{
    class SynthRunnerParameters : public BaseRunnerParameters
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

        uint num_ones;
        uint step;
        uint max_arrays;

        SynthRunnerParameters(uint image_count, uint images_read, uint block_count, uint threads_per_block, uint mask_length,
                              uint cells_count, uint address_length, uint value_length,
                              uint num_ones, uint step, uint max_arrays) :

                image_count(image_count),
                images_read(images_read),
                block_count(block_count),
                threads_per_block(threads_per_block),
                mask_length(mask_length),
                cells_count(cells_count),
                address_length(address_length),
                value_length(value_length),
                num_ones(num_ones),
                step(step),
                max_arrays(max_arrays)
        {}
    };

    class SynthRunner : public BaseRunner
    {
    public:
        report_map jaeckel(const std::string& data_path, const std::string& output_path);
        report_map labels(const std::string& data_path, const std::string& output_path);
        report_map cs_conf1(const std::string& data_path, const std::string& output_path);
        report_map cs_conf2(const std::string& data_path, const std::string& output_path);
        report_map cs_conf3(const std::string& data_path, const std::string& output_path);
        report_map cs_conf4(const std::string& data_path, const std::string& output_path);

        void set_data(bool** d) { this->data = *d; }

        void set_parameters(SynthRunnerParameters* params) { this->parameters = params; }
        SynthRunnerParameters* get_parameters() { return this->parameters; }
    private:
        bool* data;
        SynthRunnerParameters* parameters;
    };
}


#endif //synth_runner_cuh
