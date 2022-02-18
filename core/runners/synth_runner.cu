#ifndef cs2_runner_cu
#define cs2_runner_cu

#include <fstream>

#include "synth_runner.cuh"


report_map Runners::SynthRunner::jaeckel(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    auto sparse_array_indices = read_sparse_arrays<short>(parameters->num_ones, data_path);

    std::vector<bool*> sparse_arrays;
    for (auto el: sparse_array_indices)
    {
        auto sparse_array = get_sparse_array(el, parameters->address_length);
        sparse_arrays.push_back(sparse_array);
    }

    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length, parameters->cells_count, data_path);

//    SDM_JAECKEL<short, short, int> sdm_jaeckel(parameters->mask_length, parameters->address_length, parameters->value_length, parameters->cells_count,
//                                               parameters->block_count, parameters->threads_per_block,
//                                               0.0, mask_indices=mask_indices);

    SDM_LABELS<signed char, short, int> sdm_jaeckel(parameters->mask_length, parameters->address_length, parameters->value_length, parameters->cells_count,
                                               parameters->block_count, parameters->threads_per_block, ReadingType::STATISTICAL,
                                               0.0, mask_indices=mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    long long sizeof_char = sizeof(char);
    double acts_all = 0.0;
    while (all_indices.size() != parameters->knots[parameters->knots.size() - 1])
    {
        int zero_acts = 0;
        std::vector<int> indices;

        std::ofstream file_acts;
        std::string acts_str;
        acts_str.reserve(4*parameters->knots[iter]);
        while (all_indices.size() != parameters->knots[iter])
        {
            bool* value = sparse_arrays[write_index];

            int acts = sdm_jaeckel.write(value, value);
            indices.push_back(write_index);
            all_indices.push_back(write_index);

            write_index++;
            acts_str += std::to_string(acts) + ((write_index == parameters->knots[iter] - 1) ? "" : ",");
            acts_all += acts;
            zero_acts += (acts == 0);
            if (write_index % 100000 == 0)
                std::cout << "\t" << write_index << " " << get_time();
        }
        std::cout << "s=" << parameters->num_ones << " arrays=" << parameters->knots[iter] <<
        " zero_acts=" << zero_acts << " acts_avg=" << acts_all/parameters->knots[iter] << " written=" << all_indices.size() << " " << get_time();

        file_acts.open("/home/rolandw0w/Development/PhD/output/synth/jaeckel/s" + std::to_string(parameters->num_ones) +
                       "/acts_K_" + std::to_string(parameters->mask_length) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");

        file_acts.write(acts_str.c_str(), sizeof_char*acts_str.size());
        file_acts.close();

        std::string buffer;
        buffer.reserve(3*parameters->value_length*parameters->knots[iter]);

        std::string buffer_noisy;
        buffer_noisy.reserve(3*parameters->value_length*parameters->knots[iter]);
        double l1s = 0.0;
        double l1s_noisy = 0.0;
        for (auto i: all_indices)
        {
            bool* value = sparse_arrays[i];
            bool* addr = (bool*) malloc(parameters->address_length * sizeof(bool));
            for (int j = 0; j < parameters->address_length; j++)
                addr[j] = value[j];

            bool* restored = sdm_jaeckel.read(addr);
            uint l1 = hamming_distance(restored, value, parameters->value_length);
            l1s += l1;
            //file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : "";
                //file_read << restored[j] << sep;
                buffer += std::to_string(restored[j]) + sep;
            }

            // get noisy
            std::mt19937 generator(i);
            std::uniform_int_distribution<int> u_distribution(0, parameters->num_ones - 1);

            int swap_swap_index = u_distribution(generator);
            int swap_index = sparse_array_indices[i][swap_swap_index];

            addr[swap_index] = false;

            // read noisy
            bool* restored_noisy = sdm_jaeckel.read(addr);
            uint l1_noisy = hamming_distance(restored_noisy, value, parameters->value_length);
            l1s_noisy += l1_noisy;
            //file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : "";
                //file_read_noisy << restored_noisy[j] << sep;
                buffer_noisy += std::to_string(restored_noisy[j]) + sep;
            }

            // read noisy
            free(addr);
            free(restored);
            free(restored_noisy);
        }
        std::cout << "avg_l1=" << l1s / parameters->knots[iter] << " avg_l1_noisy=" << l1s_noisy / parameters->knots[iter] << " " << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/jaeckel/s" + std::to_string(parameters->num_ones) +
                       "/read_K_" + std::to_string(parameters->mask_length) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
        file_read.write(buffer.c_str(), sizeof_char*buffer.size());
        file_read.close();

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/jaeckel/s" + std::to_string(parameters->num_ones) +
                             "/read_noisy_K_" + std::to_string(parameters->mask_length) +
                             "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
        file_read_noisy.write(buffer_noisy.c_str(), sizeof_char*buffer_noisy.size());
        file_read_noisy.close();
        iter++;
    }
    for (auto sparse_array: sparse_arrays)
    {
        free(sparse_array);
    }

    return report;
}


report_map Runners::SynthRunner::labels(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    auto sparse_array_indices = read_sparse_arrays<short>(parameters->num_ones, data_path);

    std::vector<bool*> sparse_arrays;
    for (auto el: sparse_array_indices)
    {
        auto sparse_array = get_sparse_array(el, parameters->address_length);
        sparse_arrays.push_back(sparse_array);
    }

    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length, parameters->cells_count, data_path);

//    SDM_JAECKEL<short, short, int> sdm_jaeckel(parameters->mask_length, parameters->address_length, parameters->value_length, parameters->cells_count,
//                                               parameters->block_count, parameters->threads_per_block,
//                                               0.0, mask_indices=mask_indices);

    SDM_LABELS<short, short, int> sdm_jaeckel(parameters->mask_length, parameters->address_length, parameters->value_length, parameters->cells_count,
                                              parameters->block_count, parameters->threads_per_block, ReadingType::STATISTICAL,
                                              0.0, mask_indices=mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    while (all_indices.size() < parameters->max_arrays)
    {
        iter++;
        int zero_acts = 0;
        std::vector<int> indices;

        while (indices.size() != parameters->step)
        {
            bool* value = sparse_arrays[write_index];

            int acts = sdm_jaeckel.write(value, value);
            if (acts != 0)
            {
                indices.push_back(write_index);
                all_indices.push_back(write_index);
            }
            write_index++;
        }
        std::cout << "s=" << parameters->num_ones << " arrays=" << iter*parameters->step <<
                  " zero_acts=" << zero_acts << " written=" << all_indices.size() << std::endl;

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/labels/read_S_" + std::to_string(parameters->num_ones) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/labels/read_noisy_S_" + std::to_string(parameters->num_ones) +
                             "_K_" + std::to_string(parameters->mask_length) +
                             "_I_" + std::to_string(iter*parameters->step) + ".csv");
        double l1 = 0.0;
        double l1_n = 0.0;
        for (auto i: all_indices)
        {

            bool* value = sparse_arrays[i];
            bool* addr = (bool*) malloc(parameters->address_length * sizeof(bool));
            for (int j = 0; j < parameters->address_length; j++)
                addr[j] = value[j];

            bool* restored = sdm_jaeckel.read(value, addr);
            file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                l1 += (restored[j] != value[j]);
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read << restored[j] << sep;
            }

            // get noisy
            std::mt19937 generator(i);
            std::uniform_int_distribution<int> u_distribution(0, parameters->num_ones - 1);

            int swap_swap_index = u_distribution(generator);
            int swap_index = sparse_array_indices[i][swap_swap_index];

            addr[swap_index] = false;

            // read noisy
            bool* restored_noisy = sdm_jaeckel.read(value, addr);
            file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                l1_n += (restored_noisy[j] != value[j]);
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }
            // read noisy
            free(addr);
            free(restored);
            free(restored_noisy);
        }
        std::cout << "avg_l1=" << l1/all_indices.size() << " avg_l1_n=" << l1_n/all_indices.size() << std::endl;
        file_read.close();
        file_read_noisy.close();
    }
    for (auto sparse_array: sparse_arrays)
    {
        free(sparse_array);
    }

    return report;
}


report_map Runners::SynthRunner::kanerva(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    auto sparse_array_indices = read_sparse_arrays<short>(parameters->num_ones, data_path);

    std::vector<bool*> sparse_arrays;
    for (auto el: sparse_array_indices)
    {
        auto sparse_array = get_sparse_array(el, parameters->address_length);
        sparse_arrays.push_back(sparse_array);
    }

    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length, parameters->cells_count, data_path);

//    SDM_JAECKEL<short, short, int> sdm_jaeckel(parameters->mask_length, parameters->address_length, parameters->value_length, parameters->cells_count,
//                                               parameters->block_count, parameters->threads_per_block,
//                                               0.0, mask_indices=mask_indices);
    int dist = parameters->radius;
//    switch (parameters->num_ones) {
//        case 12:
//            dist = 18;
//            break;
//        case 16:
//            dist = 13;
//            break;
//        case 20:
//            dist = 17;
//            break;
//    }
    SDM_KANERVA_SPARSE<signed char, short, int> sdm_kanerva(dist, parameters->mask_length, parameters->address_length,
                                                      parameters->value_length, parameters->cells_count,
                                                      parameters->block_count, parameters->threads_per_block, mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    long long sizeof_char = sizeof(char);
    double acts_all = 0.0;
    int zero_acts = 0;
    while (all_indices.size() != parameters->knots[parameters->knots.size() - 1])
    {
        std::vector<int> indices;

        std::ofstream file_acts;
        std::string acts_str;
        acts_str.reserve(4*parameters->knots[iter]);
        while (all_indices.size() != parameters->knots[iter])
        {
            bool* value = sparse_arrays[write_index];

            int acts = sdm_kanerva.write(value, value);
            indices.push_back(write_index);
            all_indices.push_back(write_index);

            write_index++;
            acts_str += std::to_string(acts) + ((write_index == parameters->knots[iter] - 1) ? "" : ",");
            acts_all += acts;
            zero_acts += (acts == 0);
        }
        std::cout << "radius=" << dist << " s=" << parameters->num_ones << " arrays=" << parameters->knots[iter] <<
                  " zero_acts=" << zero_acts << " acts_avg=" << acts_all/parameters->knots[iter] << " written=" << all_indices.size() << " " << get_time();

        file_acts.open("/home/rolandw0w/Development/PhD/output/synth/kanerva/s" + std::to_string(parameters->num_ones) +
                       "/acts_R_" + std::to_string(dist) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");

        file_acts.write(acts_str.c_str(), sizeof_char*acts_str.size());
        file_acts.close();
        std::string buffer;
        buffer.reserve(3*parameters->value_length*parameters->knots[iter]);

        std::string buffer_noisy;
        buffer_noisy.reserve(3*parameters->value_length*parameters->knots[iter]);

        double l1s = 0.0;
        double l1s_noisy = 0.0;
        for (auto i: all_indices)
        {
            bool* value = sparse_arrays[i];
            bool* addr = (bool*) malloc(parameters->address_length * sizeof(bool));
            for (int j = 0; j < parameters->address_length; j++)
                addr[j] = value[j];

            bool* restored = sdm_kanerva.read(addr);
            uint l1 = hamming_distance(restored, value, parameters->value_length);
            l1s += l1;
            //file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : "";
                //file_read << restored[j] << sep;
                double r = restored[j];
                int r_int = (int) std::round(r);
                if (std::abs(r - r_int) < 1e-6)
                    buffer += std::to_string(restored[j]) + sep;
                else buffer += std::to_string(restored[j]) + sep;
            }

            // get noisy
            std::mt19937 generator(i);
            std::uniform_int_distribution<int> u_distribution(0, parameters->num_ones - 1);

            int swap_swap_index = u_distribution(generator);
            int swap_index = sparse_array_indices[i][swap_swap_index];

            addr[swap_index] = false;

            // read noisy
            bool* restored_noisy = sdm_kanerva.read(addr);
            uint l1_noisy = hamming_distance(restored_noisy, value, parameters->value_length);
            l1s_noisy += l1_noisy;
            //file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : "";
                //file_read_noisy << restored_noisy[j] << sep;
                buffer_noisy += std::to_string(restored_noisy[j]) + sep;
            }

            // read noisy
            free(addr);
            free(restored);
            free(restored_noisy);
        }
        std::cout << "radius=" << dist << " avg_l1=" << l1s / parameters->knots[iter] << " avg_l1_noisy=" << l1s_noisy / parameters->knots[iter] << " " << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/kanerva/s" + std::to_string(parameters->num_ones) +
                       "/read_R_" + std::to_string(dist) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
        file_read.write(buffer.c_str(), sizeof_char*buffer.size());
        file_read.close();

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/kanerva/s" + std::to_string(parameters->num_ones) +
                             "/read_noisy_R_" + std::to_string(dist) +
                             "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
        file_read_noisy.write(buffer_noisy.c_str(), sizeof_char*buffer_noisy.size());
        file_read_noisy.close();
        iter++;
    }
    for (auto sparse_array: sparse_arrays)
    {
        free(sparse_array);
    }

    return report;
}


report_map Runners::SynthRunner::cs_conf1(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    auto sparse_array_indices = read_sparse_arrays<short>(parameters->num_ones, data_path);

    int rows = parameters->address_length;
    int columns = 4*parameters->address_length;

    int num_to_read = 4 * parameters->max_arrays;

    bool* array_data = read_sparse_arrays_bool(sparse_array_indices, num_to_read, columns);
    bool* array_data_noisy = (bool*) malloc(columns*num_to_read*sizeof(bool));
    for (int i = 0; i < columns*num_to_read; i++)
    {
        array_data_noisy[i] = array_data[i];
    }

    for (int i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        std::uniform_int_distribution<int> u_distribution(0, parameters->num_ones - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
    );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    uint transformation_size = rows*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs_conf1/matrix.csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;
    SUM_TYPE* cuda_transformed_noisy;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    cuda_malloc(&cuda_transformed_noisy, transformation_size);
    cuda_memset(cuda_transformed_noisy, (SUM_TYPE)0, transformation_size);

    auto transformed = (SUM_TYPE*) malloc(transformation_size*sizeof(SUM_TYPE));
    auto transformed_noisy = (SUM_TYPE*) malloc(transformation_size*sizeof(SUM_TYPE));

    bool* cuda_data;
    cuda_malloc(&cuda_data, columns*num_to_read);
    cuda_memcpy_to_gpu(cuda_data, array_data, columns*num_to_read);

    bool* cuda_data_noisy;
    cuda_malloc(&cuda_data_noisy, columns*num_to_read);
    cuda_memcpy_to_gpu(cuda_data_noisy, array_data_noisy, columns*num_to_read);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
    );
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data_noisy, cuda_transformed_noisy, rows, columns, num_to_read, thread_count
    );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);
    cuda_memcpy_from_gpu(transformed_noisy, cuda_transformed_noisy, transformation_size);

    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);
    cuda_free(cuda_transformed_noisy);

    std::vector<SUM_TYPE*> arrays(num_to_read);
    std::vector<SUM_TYPE*> arrays_noisy(num_to_read);
    for (int i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (int j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto value_noisy = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (int j = 0; j < rows; j++)
        {
            auto val = transformed_noisy[j*num_to_read + i];
            value_noisy[j] = val;
        }
        arrays_noisy[i] = value_noisy;
    }


    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length, parameters->cells_count, data_path);

    SDM_CS1<short, short, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                             parameters->value_length, parameters->cells_count, parameters->block_count,
                                             parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    while (all_indices.size() < parameters->max_arrays)
    {
        iter++;
        std::vector<int> indices;

        while (indices.size() != parameters->step)
        {
            auto value = arrays[write_index];

            int acts = sdm.write(value, value);
            if (acts != 0)
            {
                indices.push_back(write_index);
                all_indices.push_back(write_index);
            }
            write_index++;
        }
        std::cout  << "s=" << parameters->num_ones << " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
        " I=" << iter*parameters->step << " wi=" << write_index << " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf1/read_S_" + std::to_string(parameters->num_ones) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf1/read_noisy_S_" + std::to_string(parameters->num_ones) +
                             "_K_" + std::to_string(parameters->mask_length) +
                             "_N_" + std::to_string(parameters->cells_count) +
                             "_I_" + std::to_string(iter*parameters->step) + ".csv");
        for (auto i: all_indices)
        {
            auto value = arrays[i];

            double* restored = sdm.read(value);
            file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read << restored[j] << sep;
            }
            auto addr_noisy = arrays_noisy[i];

            // read noisy
            double* restored_noisy = sdm.read(addr_noisy);
            file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }

            // read noisy
            free(restored);
            free(restored_noisy);
        }
        file_read.close();
        file_read_noisy.close();
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);
    free(transformed_noisy);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: arrays_noisy)
        free(arr);

    return report;
}


report_map Runners::SynthRunner::cs_conf2(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    auto sparse_array_indices = read_sparse_arrays<short>(parameters->num_ones, data_path);

    int rows = parameters->value_length;
    int columns = parameters->address_length;

    int num_to_read = 4 * parameters->max_arrays;

    bool* array_data = read_sparse_arrays_bool(sparse_array_indices, num_to_read, columns);
    bool* array_data_noisy = (bool*) malloc(columns*num_to_read*sizeof(bool));
    for (int i = 0; i < columns*num_to_read; i++)
    {
        array_data_noisy[i] = array_data[i];
    }

    for (int i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        std::uniform_int_distribution<int> u_distribution(0, parameters->num_ones - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
    );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    uint transformation_size = rows*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs_conf2/matrix.csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    auto transformed = (SUM_TYPE*) malloc(transformation_size*sizeof(SUM_TYPE));

    bool* cuda_data;
    cuda_malloc(&cuda_data, columns*num_to_read);
    cuda_memcpy_to_gpu(cuda_data, array_data, columns*num_to_read);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
    );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);

    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);

    std::vector<SUM_TYPE *> arrays(num_to_read);

    std::vector<bool*> addresses(num_to_read);
    std::vector<bool*> addresses_noisy(num_to_read);
    for (int i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (int j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto address = (bool*) malloc(columns*sizeof(bool));
        for (int j = 0; j < columns; j++)
        {
            auto val = array_data[i*columns + j];
            address[j] = val;
        }
        addresses[i] = address;

        auto address_noisy = (bool*) malloc(columns*sizeof(bool));
        for (int j = 0; j < columns; j++)
        {
            auto val = array_data_noisy[i*columns + j];
            address_noisy[j] = val;
        }
        addresses_noisy[i] = address_noisy;
    }


    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length, parameters->cells_count, data_path);

    SDM_CS2<short, short, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                               parameters->value_length, parameters->cells_count, parameters->block_count,
                                               parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    while (all_indices.size() < parameters->max_arrays)
    {
        iter++;
        std::vector<int> indices;

        while (indices.size() != parameters->step)
        {
            auto value = arrays[write_index];
            auto address = addresses[write_index];

            int acts = sdm.write(value, address);
            if (acts != 0)
            {
                indices.push_back(write_index);
                all_indices.push_back(write_index);
            }
            write_index++;
        }
        std::cout  << "s=" << parameters->num_ones << " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
                   " I=" << iter*parameters->step << " wi=" << write_index << " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf2/read_S_" + std::to_string(parameters->num_ones) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf2/read_noisy_S_" + std::to_string(parameters->num_ones) +
                             "_K_" + std::to_string(parameters->mask_length) +
                             "_N_" + std::to_string(parameters->cells_count) +
                             "_I_" + std::to_string(iter*parameters->step) + ".csv");
        for (auto i: all_indices)
        {
            auto address = addresses[i];
            auto address_noisy = addresses_noisy[i];

            double* restored = sdm.read(address);
            file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read << restored[j] << sep;
            }

            // read noisy
            double* restored_noisy = sdm.read(address_noisy);
            file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }

            // read noisy
            free(restored);
            free(restored_noisy);
        }
        file_read.close();
        file_read_noisy.close();
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: addresses)
        free(arr);
    for (auto arr: addresses_noisy)
        free(arr);

    return report;
}


report_map Runners::SynthRunner::cs_conf3(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    auto sparse_array_indices = read_sparse_arrays<short>(parameters->num_ones, data_path);

    int rows = parameters->address_length;
    int columns = 600;//parameters->address_length;

    int num_to_read = 4 * parameters->max_arrays;

    bool* array_data = read_sparse_arrays_bool(sparse_array_indices, num_to_read, columns);
    bool* array_data_noisy = (bool*) malloc(columns*num_to_read*sizeof(bool));
    for (int i = 0; i < columns*num_to_read; i++)
    {
        array_data_noisy[i] = array_data[i];
    }

    for (int i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        std::uniform_int_distribution<int> u_distribution(0, parameters->num_ones - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
    );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    uint transformation_size = rows*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs_conf3/s" + std::to_string(parameters->num_ones) +
            "/matrix_" + std::to_string(parameters->address_length) + ".csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;
    SUM_TYPE* cuda_transformed_noisy;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    cuda_malloc(&cuda_transformed_noisy, transformation_size);
    cuda_memset(cuda_transformed_noisy, (SUM_TYPE)0, transformation_size);

    auto transformed = (SUM_TYPE*) malloc(transformation_size*sizeof(SUM_TYPE));
    auto transformed_noisy = (SUM_TYPE*) malloc(transformation_size*sizeof(SUM_TYPE));

    bool* cuda_data;
    cuda_malloc(&cuda_data, columns*num_to_read);
    cuda_memcpy_to_gpu(cuda_data, array_data, columns*num_to_read);

    bool* cuda_data_noisy;
    cuda_malloc(&cuda_data_noisy, columns*num_to_read);
    cuda_memcpy_to_gpu(cuda_data_noisy, array_data_noisy, columns*num_to_read);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
    );
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data_noisy, cuda_transformed_noisy, rows, columns, num_to_read, thread_count
    );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);
    cuda_memcpy_from_gpu(transformed_noisy, cuda_transformed_noisy, transformation_size);

    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);
    cuda_free(cuda_transformed_noisy);

    std::vector<SUM_TYPE*> arrays(num_to_read);
    std::vector<SUM_TYPE*> arrays_noisy(num_to_read);
    for (int i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (int j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto value_noisy = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (int j = 0; j < rows; j++)
        {
            auto val = transformed_noisy[j*num_to_read + i];
            value_noisy[j] = val;
        }
        arrays_noisy[i] = value_noisy;
    }

    auto mask_indices = read_mask_indices_s<short>(parameters->mask_length, parameters->address_length,
                                                   parameters->cells_count, data_path, parameters->num_ones);

    SDM_CS1<short, short, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                               parameters->value_length, parameters->cells_count, parameters->block_count,
                                               parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    while (all_indices.size() < parameters->max_arrays)
    {
        iter++;
        std::vector<int> indices;

        while (indices.size() != parameters->step)
        {
            auto value = arrays[write_index];

            int acts = sdm.write(value, value);
            if (acts != 0)
            {
                indices.push_back(write_index);
                all_indices.push_back(write_index);
            }
            write_index++;
        }
        std::cout << "Finished writing | " <<
            "s=" << parameters->num_ones << " coef=" << parameters->value_length/parameters->num_ones << " m=" << parameters->value_length <<
            " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
            " I=" << iter*parameters->step << " wi=" << write_index << " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf3/s" + std::to_string(parameters->num_ones) +
                        "/read_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf3/s" + std::to_string(parameters->num_ones) +
                             "/read_noisy_m_" + std::to_string(parameters->value_length) +
                             "_K_" + std::to_string(parameters->mask_length) +
                             "_N_" + std::to_string(parameters->cells_count) +
                             "_I_" + std::to_string(iter*parameters->step) + ".csv");
        for (auto i: all_indices)
        {
            auto value = arrays[i];

            double* restored = sdm.read(value);
            file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read << restored[j] << sep;
            }
            auto addr_noisy = arrays_noisy[i];

            // read noisy
            double* restored_noisy = sdm.read(addr_noisy);
            file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }

            // read noisy
            free(restored);
            free(restored_noisy);
        }
        file_read.close();
        file_read_noisy.close();
        std::cout  << "Finished reading | " <<
            "s=" << parameters->num_ones << " coef=" << parameters->value_length/parameters->num_ones << " m=" << parameters->value_length <<
            " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
            " I=" << iter*parameters->step << " wi=" << write_index << " time=" << get_time();
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);
    free(transformed_noisy);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: arrays_noisy)
        free(arr);

    return report;
}


report_map Runners::SynthRunner::cs_conf4(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    auto sparse_array_indices = read_sparse_arrays<short>(parameters->num_ones, data_path);

    int rows = parameters->value_length;
    int columns = parameters->address_length;

    long long num_to_read = parameters->knots[parameters->knots.size() - 1];

    bool* array_data = read_sparse_arrays_bool(sparse_array_indices, num_to_read, columns);
    long long l = num_to_read * columns;
    bool* array_data_noisy = (bool*) malloc(l*sizeof(bool));
    for (long long i = 0; i < l; i++)
    {
        array_data_noisy[i] = array_data[i];
    }

    for (long long i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        std::uniform_int_distribution<int> u_distribution(0, parameters->num_ones - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
    );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    long long transformation_size = ((long long) rows)*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs_conf4/s" + std::to_string(parameters->num_ones) +
                                    "/matrix_" + std::to_string(parameters->value_length) + ".csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    long long t = transformation_size*sizeof(SUM_TYPE);
    auto transformed = (SUM_TYPE*) malloc(t);

    long long data_len = num_to_read * columns;

    bool* cuda_data;
    cuda_malloc(&cuda_data, data_len);
    cuda_memcpy_to_gpu(cuda_data, array_data, data_len);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
    );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);

    cuda_free(cuda_data);
    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);

    std::vector<SUM_TYPE*> arrays(num_to_read);

    std::vector<bool*> addresses(num_to_read);
    std::vector<bool*> addresses_noisy(num_to_read);
    for (long long i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (long long j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto address = (bool*) malloc(columns*sizeof(bool));
        for (long long j = 0; j < columns; j++)
        {
            auto val = array_data[i*columns + j];
            address[j] = val;
        }
        addresses[i] = address;

        auto address_noisy = (bool*) malloc(columns*sizeof(bool));
        for (long long j = 0; j < columns; j++)
        {
            auto val = array_data_noisy[i*columns + j];
            address_noisy[j] = val;
        }
        addresses_noisy[i] = address_noisy;
    }

    auto mask_indices = read_mask_indices_s<short>(parameters->mask_length, parameters->address_length,
                                                   parameters->cells_count, data_path, parameters->num_ones);

    SDM_CS2<signed char, short, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                               parameters->value_length, parameters->cells_count, parameters->block_count,
                                               parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);
    std::cout << "Started processing | " << get_time();

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    double acts_avg = 0.0;
    double zero_acts = 0.0;
    long long sizeof_char = sizeof(char);
    while (all_indices.size() != parameters->knots[parameters->knots.size() - 1])
    {
        std::ofstream file_acts;
        std::string acts_str;
        std::vector<int> indices;

        while (all_indices.size() != parameters->knots[iter])
        {
            auto value = arrays[write_index];
            auto address = addresses[write_index];

            int acts = sdm.write(value, address);
            indices.push_back(write_index);
            all_indices.push_back(write_index);
            write_index++;
            acts_avg += acts;
            acts_str += std::to_string(acts) + ((write_index == parameters->knots[iter] - 1) ? "" : ",");
            zero_acts += (acts == 0);
            if (write_index % 10000 == 0)
                std::cout << "\t" << write_index << " " << get_time();
        }

        file_acts.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s" + std::to_string(parameters->num_ones) +
                       "/acts_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");

        file_acts.write(acts_str.c_str(), sizeof_char*acts_str.size());
        file_acts.close();
        std::cout << std::endl << "Finished writing |" <<
                  " s=" << parameters->num_ones <<
                  " coef=" << parameters->value_length/parameters->num_ones <<
                  " m=" << parameters->value_length <<
                  " N=" << parameters->cells_count <<
                  " K=" << parameters->mask_length <<
                  " I=" << parameters->knots[iter] <<
                  " wi=" << write_index <<
                  " acts_avg=" << acts_avg / all_indices.size() <<
                  " zero_acts=" << zero_acts <<
                  " time=" << get_time();

//        std::ofstream file_read;
//        std::ofstream whisker_box_file;

//        std::string buffer;
//        buffer.reserve((long long) 8 * 1024 * 1024 * 1024);
//        std::string buffer_noisy;
//        buffer_noisy.reserve((long long) 8 * 1024 * 1024 * 1024);
//        std::string buffer_wb;
//        buffer_wb.reserve( 8*1024);

//        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s" + std::to_string(parameters->num_ones) +
//                       "/read_m_" + std::to_string(parameters->value_length) +
//                       "_K_" + std::to_string(parameters->mask_length) +
//                       "_N_" + std::to_string(parameters->cells_count) +
//                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");

//        std::ofstream file_read_noisy;
//        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s" + std::to_string(parameters->num_ones) +
//                             "/read_noisy_m_" + std::to_string(parameters->value_length) +
//                             "_K_" + std::to_string(parameters->mask_length) +
//                             "_N_" + std::to_string(parameters->cells_count) +
//                             "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
//
//        whisker_box_file.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s" + std::to_string(parameters->num_ones) +
//                           "/wb_m_" + std::to_string(parameters->value_length) +
//                           "_K_" + std::to_string(parameters->mask_length) +
//                           "_N_" + std::to_string(parameters->cells_count) +
//                           "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
//        for (auto i: all_indices)
//        {
//            auto address = addresses[i];
//            auto address_noisy = addresses_noisy[i];
//
//            double* restored = sdm.read(address);
//            file_read << i << "->";
//            for (int j = 0; j < parameters->value_length; j++)
//            {
//                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
//                file_read << restored[j] << sep;
//            }
//            buffer += std::to_string(i) + "->";
//            for (int j = 0; j < parameters->value_length; j++)
//            {
//                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
//                double r = restored[j];
//                int r_int = (int) std::round(r);
//                if (std::abs(r - r_int) < 1e-6)
//                    buffer += std::to_string(r_int) + sep;
//                else buffer += std::to_string(restored[j]).substr(0, 7) + sep;
//            }
//
//            // read noisy
//            double* restored_noisy = sdm.read(address_noisy);
//            file_read_noisy << i << "->";
//            for (int j = 0; j < parameters->value_length; j++)
//            {
//                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
//                file_read_noisy << restored_noisy[j] << sep;
//            }
//            buffer_noisy += std::to_string(i) + "->";
//            for (int j = 0; j < parameters->value_length; j++)
//            {
//                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
//                double r = restored_noisy[j];
//                int r_int = (int) std::round(r);
//                if (std::abs(r - r_int) < 1e-6)
//                    buffer_noisy += std::to_string(r_int) + sep;
//                else buffer_noisy += std::to_string(restored_noisy[j]).substr(0, 7) + sep;
//            }
//            if (i % 100000 == 0)
//                std::cout << "\t" << i << " " << get_time();
//
//            // read noisy
//            free(restored);
//            free(restored_noisy);
//        }
//        std::cout << "Started writing to files | size=" << buffer.size() << " noisy_size=" << buffer_noisy.size() << " time=" << get_time();
        //file_read.write(buffer.c_str(), sizeof_char*buffer.size());
        //file_read_noisy.write(buffer_noisy.c_str(), sizeof_char*buffer_noisy.size());
        //std::cout << "Finished writing to files | size=" << buffer.size() << " noisy_size=" << buffer_noisy.size() << " time=" << get_time();

        //file_read.close();
        //file_read_noisy.close();
//        buffer = "";
//        buffer_noisy = "";
//        std::cout  << "Finished reading | " <<
//                   "s=" << parameters->num_ones << " coef=" << parameters->value_length/parameters->num_ones << " m=" << parameters->value_length <<
//                   " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
//                   " I=" << parameters->knots[iter] << " wi=" << write_index << " time=" << get_time();
//
//        std::cout << "Started calculating whisker boxes: " << get_time();
//        auto wbs = sdm.get_whisker_boxes();
//        std::cout << "Finished calculating whisker boxes: " << get_time();
//        for (auto wb: wbs)
//        {
//            for (int k = 0; k < 5; k++)
//            {
//                auto sep = (k == 4) ? "\n" : ",";
//                buffer_wb += std::to_string(wb[k]);
//                buffer_wb += sep;
//            }
//        }
//        whisker_box_file.write(buffer_wb.c_str(), sizeof_char*buffer_wb.size());
//        std::cout << "Whisker boxes saved: " << get_time();
//        whisker_box_file.close();
        //wbs.clear();

        iter++;
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: addresses)
        free(arr);
    for (auto arr: addresses_noisy)
        free(arr);

    return report;
}


report_map Runners::SynthRunner::cs_conf4_mixed(const std::string &data_path, const std::string &output_path)
{
    report_map report;

    int n = 50000;

    auto sparse_array_indices_12 = read_sparse_arrays<short>(12, data_path, n);
    auto sparse_array_indices_16 = read_sparse_arrays<short>(16, data_path, n);
    auto sparse_array_indices_20 = read_sparse_arrays<short>(20, data_path, n);

    auto sparse_array_indices = std::vector<std::vector<short>>(3*n);
    for (int i = 0; i < n; i++)
    {
        sparse_array_indices[3*i] = sparse_array_indices_12[i];
        sparse_array_indices[3*i + 1] = sparse_array_indices_16[i];
        sparse_array_indices[3*i + 2] = sparse_array_indices_20[i];
    }

    int rows = parameters->value_length;
    int columns = parameters->address_length;

    long long num_to_read = parameters->knots[parameters->knots.size() - 1];

    bool* array_data = read_sparse_arrays_bool(sparse_array_indices, num_to_read, columns);
    long long l = num_to_read * columns;
    bool* array_data_noisy = (bool*) malloc(l*sizeof(bool));
    for (long long i = 0; i < l; i++)
    {
        array_data_noisy[i] = array_data[i];
    }

    for (long long i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        uint s = 4 * (3 + (i % 3));
        std::uniform_int_distribution<int> u_distribution(0, s - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
    );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    long long transformation_size = ((long long) rows)*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs_conf4_mixed//matrix_" + std::to_string(parameters->value_length) + ".csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    long long t = transformation_size*sizeof(SUM_TYPE);
    auto transformed = (SUM_TYPE*) malloc(t);

    long long data_len = num_to_read * columns;

    bool* cuda_data;
    cuda_malloc(&cuda_data, data_len);
    cuda_memcpy_to_gpu(cuda_data, array_data, data_len);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
    );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);

    cuda_free(cuda_data);
    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);

    std::vector<SUM_TYPE*> arrays(num_to_read);

    std::vector<bool*> addresses(num_to_read);
    std::vector<bool*> addresses_noisy(num_to_read);
    for (long long i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (long long j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto address = (bool*) malloc(columns*sizeof(bool));
        for (long long j = 0; j < columns; j++)
        {
            auto val = array_data[i*columns + j];
            address[j] = val;
        }
        addresses[i] = address;

        auto address_noisy = (bool*) malloc(columns*sizeof(bool));
        for (long long j = 0; j < columns; j++)
        {
            auto val = array_data_noisy[i*columns + j];
            address_noisy[j] = val;
        }
        addresses_noisy[i] = address_noisy;
    }

    auto mask_indices = read_mask_indices_s<short>(parameters->mask_length, parameters->address_length,
                                                   parameters->cells_count, data_path, parameters->num_ones);

    SDM_CS2<signed char, short, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                                     parameters->value_length, parameters->cells_count, parameters->block_count,
                                                     parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);
    std::cout << "Started processing | " << get_time();

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    double acts_avg = 0.0;
    double zero_acts = 0.0;
    long long sizeof_char = sizeof(char);
    while (all_indices.size() != parameters->knots[parameters->knots.size() - 1])
    {
        std::ofstream file_acts;
        std::string acts_str;
        std::vector<int> indices;

        while (all_indices.size() != parameters->knots[iter])
        {
            auto value = arrays[write_index];
            auto address = addresses[write_index];

            int acts = sdm.write(value, address);
            indices.push_back(write_index);
            all_indices.push_back(write_index);
            write_index++;
            acts_avg += acts;
            acts_str += std::to_string(acts) + ((write_index == parameters->knots[iter] - 1) ? "" : ",");
            zero_acts += (acts == 0);
            if (write_index % 1000 == 0)
                std::cout << "\t" << write_index << " " << get_time();
        }

        file_acts.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf4_mixed/acts_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");

        file_acts.write(acts_str.c_str(), sizeof_char*acts_str.size());
        file_acts.close();
        std::cout << std::endl << "Finished writing |" <<
                  " m=" << parameters->value_length <<
                  " N=" << parameters->cells_count <<
                  " K=" << parameters->mask_length <<
                  " I=" << parameters->knots[iter] <<
                  " wi=" << write_index <<
                  " acts_avg=" << acts_avg / all_indices.size() <<
                  " zero_acts=" << zero_acts <<
                  " time=" << get_time();

        std::ofstream file_read;
        std::ofstream whisker_box_file;

        std::string buffer;
        buffer.reserve((long long) 8 * 1024 * 1024 * 1024);
        std::string buffer_noisy;
        buffer_noisy.reserve((long long) 8 * 1024 * 1024 * 1024);
        std::string buffer_wb;
        buffer_wb.reserve( 8*1024);

        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf4_mixed/read_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf4_mixed/read_noisy_m_" + std::to_string(parameters->value_length) +
                             "_K_" + std::to_string(parameters->mask_length) +
                             "_N_" + std::to_string(parameters->cells_count) +
                             "_I_" + std::to_string(parameters->knots[iter]) + ".csv");

        whisker_box_file.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf4_mixed/wb_m_" + std::to_string(parameters->value_length) +
                           "_K_" + std::to_string(parameters->mask_length) +
                           "_N_" + std::to_string(parameters->cells_count) +
                           "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
        for (auto i: all_indices)
        {
            auto address = addresses[i];
            auto address_noisy = addresses_noisy[i];

            double* restored = sdm.read(address);
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                double r = restored[j];
                int r_int = (int) std::round(r);
                if (std::abs(r - r_int) < 1e-6)
                    buffer += std::to_string(r_int) + sep;
                else buffer += std::to_string(restored[j]).substr(0, 7) + sep;
            }

            // read noisy
            double* restored_noisy = sdm.read(address_noisy);
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                double r = restored_noisy[j];
                int r_int = (int) std::round(r);
                if (std::abs(r - r_int) < 1e-6)
                    buffer_noisy += std::to_string(r_int) + sep;
                else buffer_noisy += std::to_string(restored_noisy[j]).substr(0, 7) + sep;
            }
            if (i % 1000 == 0)
                std::cout << "\t" << i << " " << get_time();

            // read noisy
            free(restored);
            free(restored_noisy);
        }
        std::cout << "Started writing to files | size=" << buffer.size() << " noisy_size=" << buffer_noisy.size() << " time=" << get_time();
        file_read.write(buffer.c_str(), sizeof_char*buffer.size());
        file_read_noisy.write(buffer_noisy.c_str(), sizeof_char*buffer_noisy.size());
        std::cout << "Finished writing to files | size=" << buffer.size() << " noisy_size=" << buffer_noisy.size() << " time=" << get_time();

        file_read.close();
        file_read_noisy.close();
        buffer = "";
        buffer_noisy = "";
        std::cout  << "Finished reading | " << " m=" << parameters->value_length <<
                   " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
                   " I=" << parameters->knots[iter] << " wi=" << write_index << " time=" << get_time();

        std::cout << "Started calculating whisker boxes: " << get_time();
        auto wbs = sdm.get_whisker_boxes();
        std::cout << "Finished calculating whisker boxes: " << get_time();
        for (auto wb: wbs)
        {
            for (int k = 0; k < 5; k++)
            {
                auto sep = (k == 4) ? "\n" : ",";
                buffer_wb += std::to_string(wb[k]);
                buffer_wb += sep;
            }
        }
        whisker_box_file.write(buffer_wb.c_str(), sizeof_char*buffer_wb.size());
        std::cout << "Whisker boxes saved: " << get_time();
        whisker_box_file.close();
        wbs.clear();

        iter++;
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: addresses)
        free(arr);
    for (auto arr: addresses_noisy)
        free(arr);

    return report;
}


report_map Runners::SynthRunner::cs_conf_reverse(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    auto sparse_array_indices = read_sparse_arrays<short>(parameters->num_ones, data_path);

    int rows = parameters->value_length;
    int columns = parameters->address_length;

    long long num_to_read = parameters->knots[parameters->knots.size() - 1];

    bool* array_data = read_sparse_arrays_bool(sparse_array_indices, num_to_read, columns);
    long long l = num_to_read * columns;
    bool* array_data_noisy = (bool*) malloc(l*sizeof(bool));
    for (long long i = 0; i < l; i++)
    {
        array_data_noisy[i] = array_data[i];
    }

    for (long long i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        std::uniform_int_distribution<int> u_distribution(0, parameters->num_ones - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
    );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    long long transformation_size = ((long long) rows)*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs_conf_reverse/s" + std::to_string(parameters->num_ones) +
                                    "/matrix_" + std::to_string(parameters->value_length) + ".csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    long long t = transformation_size*sizeof(SUM_TYPE);
    auto transformed = (SUM_TYPE*) malloc(t);

    long long data_len = num_to_read * columns;

    bool* cuda_data;
    cuda_malloc(&cuda_data, data_len);
    cuda_memcpy_to_gpu(cuda_data, array_data, data_len);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
    );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);

    cuda_free(cuda_data);
    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);

    std::vector<SUM_TYPE*> arrays(num_to_read);

    std::vector<bool*> addresses(num_to_read);
    std::vector<bool*> addresses_noisy(num_to_read);
    for (long long i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (long long j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto address = (bool*) malloc(columns*sizeof(bool));
        for (long long j = 0; j < columns; j++)
        {
            auto val = array_data[i*columns + j];
            address[j] = val;
        }
        addresses[i] = address;

        auto address_noisy = (bool*) malloc(columns*sizeof(bool));
        for (long long j = 0; j < columns; j++)
        {
            auto val = array_data_noisy[i*columns + j];
            address_noisy[j] = val;
        }
        addresses_noisy[i] = address_noisy;
    }

    auto mask_indices = read_mask_indices_s_binary<long long>(
            parameters->mask_length, parameters->address_length,parameters->cells_count, data_path, parameters->num_ones);
//    for (int i = 0; i < 75; i++)
//    {
//        std::cout << mask_indices[i] <<" ";
//    }
//    std::cout << std::endl;
//    for (int i = 0; i < 75; i++)
//    {
//        auto a = to_bits(mask_indices[i], 8);
//        for (int j = 0; j < 8; j++)
//            std::cout << a[j];
//        std::cout << std::endl;
//        for(int j = 7; j >= 0; j--)
//            std::cout << ((mask_indices[i] & (1 << (j))) ? 1 : 0);
//        std::cout << std::endl;
//        std::cout << std::endl;
//        free(a);
//    }
//    std::cout << std::endl;

    SDM_CS_REVERSE<signed char, char, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                                           parameters->value_length, parameters->cells_count, parameters->block_count,
                                                           parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);
    std::cout << "Started processing | " << get_time();

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    double acts_avg = 0.0;
    double zero_acts = 0.0;
    long long sizeof_char = sizeof(char);
    while (all_indices.size() != parameters->knots[parameters->knots.size() - 1])
    {
        std::ofstream file_acts;
        std::string acts_str;
        std::vector<int> indices;

        while (all_indices.size() != parameters->knots[iter])
        {
            auto value = arrays[write_index];
            auto address = addresses[write_index];

            int acts = sdm.write(value, address);
            indices.push_back(write_index);
            all_indices.push_back(write_index);
            write_index++;
            acts_avg += acts;
            acts_str += std::to_string(acts) + ((write_index == parameters->knots[iter] - 1) ? "" : ",");
            zero_acts += (acts == 0);
            if (write_index % 1000 == 0)
                std::cout << "\t" << write_index << " " << get_time();
        }

        file_acts.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse/s" + std::to_string(parameters->num_ones) +
                       "/acts_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");

        file_acts.write(acts_str.c_str(), sizeof_char*acts_str.size());
        file_acts.close();
        std::cout << std::endl << "Finished writing |" <<
                  " s=" << parameters->num_ones <<
                  " coef=" << parameters->value_length/parameters->num_ones <<
                  " m=" << parameters->value_length <<
                  " N=" << parameters->cells_count <<
                  " K=" << parameters->mask_length <<
                  " I=" << parameters->knots[iter] <<
                  " wi=" << write_index <<
                  " acts_avg=" << acts_avg / all_indices.size() <<
                  " zero_acts=" << zero_acts <<
                  " time=" << get_time();

        std::string buffer;
        buffer.reserve((long long) 8 * 1024 * 1024 * 1024);
        std::string buffer_noisy;
        buffer_noisy.reserve((long long) 8 * 1024 * 1024 * 1024);
        std::string buffer_wb;
        buffer_wb.reserve( 8*1024);

        for (auto i: all_indices)
        {
            auto address = addresses[i];
            auto address_noisy = addresses_noisy[i];

            double* restored = sdm.read(address);
//            file_read << i << "->";
//            for (int j = 0; j < parameters->value_length; j++)
//            {
//                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
//                file_read << restored[j] << sep;
//            }
//            buffer += std::to_string(i) + "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                double r = restored[j];
                int r_int = (int) std::round(r);
                if (std::abs(r - r_int) < 1e-6)
                    buffer += std::to_string(r_int) + sep;
                else buffer += std::to_string(restored[j]).substr(0, 7) + sep;
            }

            // read noisy
            double* restored_noisy = sdm.read(address_noisy);
//            file_read_noisy << i << "->";
//            for (int j = 0; j < parameters->value_length; j++)
//            {
//                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
//                file_read_noisy << restored_noisy[j] << sep;
//            }
//            buffer_noisy += std::to_string(i) + "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                double r = restored_noisy[j];
                int r_int = (int) std::round(r);
                if (std::abs(r - r_int) < 1e-6)
                    buffer_noisy += std::to_string(r_int) + sep;
                else buffer_noisy += std::to_string(restored_noisy[j]).substr(0, 7) + sep;
            }
            if (i % 1000 == 0)
                std::cout << "\t" << i << " " << get_time();

            // read noisy
            free(restored);
            free(restored_noisy);
        }
        std::cout << "Started writing to files | size=" << buffer.size() << " noisy_size=" << buffer_noisy.size() << " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse/s" + std::to_string(parameters->num_ones) +
                       "/read_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
        file_read.write(buffer.c_str(), sizeof_char*buffer.size());

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse/s" + std::to_string(parameters->num_ones) +
                             "/read_noisy_m_" + std::to_string(parameters->value_length) +
                             "_K_" + std::to_string(parameters->mask_length) +
                             "_N_" + std::to_string(parameters->cells_count) +
                             "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
        file_read_noisy.write(buffer_noisy.c_str(), sizeof_char*buffer_noisy.size());
        std::cout << "Finished writing to files | size=" << buffer.size() << " noisy_size=" << buffer_noisy.size() << " time=" << get_time();

        file_read.close();
        file_read_noisy.close();
        buffer = "";
        buffer_noisy = "";
        std::cout  << "Finished reading | " <<
                   "s=" << parameters->num_ones << " coef=" << parameters->value_length/parameters->num_ones << " m=" << parameters->value_length <<
                   " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
                   " I=" << parameters->knots[iter] << " wi=" << write_index << " time=" << get_time();

        std::cout << "Started calculating whisker boxes: " << get_time();
        auto wbs = sdm.get_whisker_boxes();
        std::cout << "Finished calculating whisker boxes: " << get_time();
        for (auto wb: wbs)
        {
            for (int k = 0; k < 5; k++)
            {
                auto sep = (k == 4) ? "\n" : ",";
                buffer_wb += std::to_string(wb[k]);
                buffer_wb += sep;
            }
        }

        std::ofstream whisker_box_file;
        whisker_box_file.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse/s" + std::to_string(parameters->num_ones) +
                              "/wb_m_" + std::to_string(parameters->value_length) +
                              "_K_" + std::to_string(parameters->mask_length) +
                              "_N_" + std::to_string(parameters->cells_count) +
                              "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
        whisker_box_file.write(buffer_wb.c_str(), sizeof_char*buffer_wb.size());
        std::cout << "Whisker boxes saved: " << get_time();
        whisker_box_file.close();
        wbs.clear();

        iter++;
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: addresses)
        free(arr);
    for (auto arr: addresses_noisy)
        free(arr);

    return report;
}


report_map Runners::SynthRunner::cs_conf_reverse_nat(const std::string &data_path, const std::string &output_path)
{
    report_map report;

    std::string dp = data_path;
    bool* d = get_cs1(600, 12000, dp);

    int c = 0;

    std::vector<std::vector<short>> sparse_array_indices;
    std::vector<bool*> sparse_arrays;
    for (int i = 0; i < 12000; i++)
    {
        int ones = 0;
        std::vector<short> sparse_array_inds;
        bool* sparse_array = (bool*) malloc(parameters->address_length * sizeof(bool));
        for (short j = 0; j < (short) parameters->address_length; j += 1)
        {
            bool b = d[i*600 + j];
            sparse_array[j] = b;

            if (b)
            {
                sparse_array_inds.push_back(j);
                ones += 1;
            }
        }
        if (ones >= 4)
        {
            c += 1;
            sparse_array_indices.push_back(sparse_array_inds);
            sparse_arrays.push_back(sparse_array);
        }
        else {
            free(sparse_array);
        }
        if (sparse_arrays.size() == parameters->max_arrays)
            break;
    }

    int rows = parameters->value_length;
    int columns = parameters->address_length;

    long long num_to_read = parameters->knots[parameters->knots.size() - 1];

    bool* array_data = read_sparse_arrays_bool(sparse_array_indices, num_to_read, columns);
    long long l = num_to_read * columns;
    bool* array_data_noisy = (bool*) malloc(l*sizeof(bool));
    for (long long i = 0; i < l; i++)
    {
        array_data_noisy[i] = array_data[i];
    }

    for (long long i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        const int sz = sparse_array_indices[i].size();
        std::uniform_int_distribution<int> u_distribution(0, sz - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
    );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    long long transformation_size = ((long long) rows)*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs_conf_reverse_nat/matrix_" + std::to_string(parameters->value_length) + ".csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    long long t = transformation_size*sizeof(SUM_TYPE);
    auto transformed = (SUM_TYPE*) malloc(t);

    long long data_len = num_to_read * columns;

    bool* cuda_data;
    cuda_malloc(&cuda_data, data_len);
    cuda_memcpy_to_gpu(cuda_data, array_data, data_len);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
    );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);

    cuda_free(cuda_data);
    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);

    std::vector<SUM_TYPE*> arrays(num_to_read);

    std::vector<bool*> addresses(num_to_read);
    std::vector<bool*> addresses_noisy(num_to_read);
    for (long long i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (long long j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto address = (bool*) malloc(columns*sizeof(bool));
        for (long long j = 0; j < columns; j++)
        {
            auto val = array_data[i*columns + j];
            address[j] = val;
        }
        addresses[i] = address;

        auto address_noisy = (bool*) malloc(columns*sizeof(bool));
        for (long long j = 0; j < columns; j++)
        {
            auto val = array_data_noisy[i*columns + j];
            address_noisy[j] = val;
        }
        addresses_noisy[i] = address_noisy;
    }

    auto mask_indices = read_mask_indices_s_binary<long long>(
            parameters->mask_length, parameters->address_length,parameters->cells_count, data_path, 0);
//    for (int i = 0; i < 75; i++)
//    {
//        std::cout << mask_indices[i] <<" ";
//    }
//    std::cout << std::endl;
//    for (int i = 0; i < 75; i++)
//    {
//        auto a = to_bits(mask_indices[i], 8);
//        for (int j = 0; j < 8; j++)
//            std::cout << a[j];
//        std::cout << std::endl;
//        for(int j = 7; j >= 0; j--)
//            std::cout << ((mask_indices[i] & (1 << (j))) ? 1 : 0);
//        std::cout << std::endl;
//        std::cout << std::endl;
//        free(a);
//    }
//    std::cout << std::endl;

    SDM_CS_REVERSE<signed char, char, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                                           parameters->value_length, parameters->cells_count, parameters->block_count,
                                                           parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);
    std::cout << "Started processing | " << get_time();

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    double acts_avg = 0.0;
    double zero_acts = 0.0;
    long long sizeof_char = sizeof(char);
    while (all_indices.size() != parameters->knots[parameters->knots.size() - 1])
    {
        std::ofstream file_acts;
        std::string acts_str;
        std::vector<int> indices;

        while (all_indices.size() != parameters->knots[iter])
        {
            auto value = arrays[write_index];
            auto address = addresses[write_index];

            int acts = sdm.write(value, address);
            indices.push_back(write_index);
            all_indices.push_back(write_index);
            write_index++;
            acts_avg += acts;
            acts_str += std::to_string(acts) + ((write_index == parameters->knots[iter] - 1) ? "" : ",");
            zero_acts += (acts == 0);
//            if (write_index % 1000 == 0)
//                std::cout << "\t" << write_index << " " << get_time();
        }

        file_acts.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse_nat/acts_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");

        file_acts.write(acts_str.c_str(), sizeof_char*acts_str.size());
        file_acts.close();
        std::cout << "Finished writing |" <<
                  " m=" << parameters->value_length <<
                  " N=" << parameters->cells_count <<
                  " K=" << parameters->mask_length <<
                  " I=" << parameters->knots[iter] <<
                  " wi=" << write_index <<
                  " acts_avg=" << acts_avg / all_indices.size() <<
                  " zero_acts=" << zero_acts <<
                  " time=" << get_time();

        std::string buffer;
        buffer.reserve((long long) 8 * 1024 * 1024 * 1024);
        std::string buffer_noisy;
        buffer_noisy.reserve((long long) 8 * 1024 * 1024 * 1024);

        for (auto i: all_indices)
        {
//            if (i != 5)
//                continue;
            auto address = addresses[i];
            auto address_noisy = addresses_noisy[i];

            double* restored = sdm.read(address);
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                double r = restored[j];
                int r_int = (int) std::round(r);
                if (std::abs(r - r_int) < 1e-6)
                    buffer += std::to_string(r_int) + sep;
                else buffer += std::to_string(restored[j]).substr(0, 7) + sep;
            }

            // read noisy
            double* restored_noisy = sdm.read(address_noisy);
//            if (restored_noisy[0] != restored_noisy[0])
//            {
//                std::cout << std::endl;
//                for (int ind = 0; ind < 600; ind++)
//                    std::cout << address[ind];
//                std::cout << std::endl;
//                for (int ind = 0; ind < 600; ind++)
//                    std::cout << address_noisy[ind];
//                std::cout << std::endl;
//                std::cout << "";
//            }
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                double r = restored_noisy[j];
                int r_int = (int) std::round(r);
                if (std::abs(r - r_int) < 1e-6)
                    buffer_noisy += std::to_string(r_int) + sep;
                else buffer_noisy += std::to_string(restored_noisy[j]).substr(0, 7) + sep;
            }
//            if (i % 1000 == 0)
//                std::cout << "\t" << i << " " << get_time();

            // read noisy
            free(restored);
            free(restored_noisy);
        }
        //std::cout << "Started writing to files | size=" << buffer.size() << " noisy_size=" << buffer_noisy.size() << " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse_nat/read_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
        file_read.write(buffer.c_str(), sizeof_char*buffer.size());

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse_nat/read_noisy_m_" + std::to_string(parameters->value_length) +
                             "_K_" + std::to_string(parameters->mask_length) +
                             "_N_" + std::to_string(parameters->cells_count) +
                             "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
        file_read_noisy.write(buffer_noisy.c_str(), sizeof_char*buffer_noisy.size());
        //std::cout << "Finished writing to files | size=" << buffer.size() << " noisy_size=" << buffer_noisy.size() << " time=" << get_time();

        file_read.close();
        file_read_noisy.close();
        buffer = "";
        buffer_noisy = "";
        std::cout << "Finished reading |" << " m=" << parameters->value_length <<
                   " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
                   " I=" << parameters->knots[iter] << " wi=" << write_index << " time=" << get_time();

//        std::cout << "Started calculating whisker boxes: " << get_time();
//        std::string buffer_wb;
//        buffer_wb.reserve( 8*1024);
//        auto wbs = sdm.get_whisker_boxes();
//        std::cout << "Finished calculating whisker boxes: " << get_time();
//        for (auto wb: wbs)
//        {
//            for (int k = 0; k < 5; k++)
//            {
//                auto sep = (k == 4) ? "\n" : ",";
//                buffer_wb += std::to_string(wb[k]);
//                buffer_wb += sep;
//            }
//        }
//
//        std::ofstream whisker_box_file;
//        whisker_box_file.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf_reverse_nat/wb_m_" + std::to_string(parameters->value_length) +
//                              "_K_" + std::to_string(parameters->mask_length) +
//                              "_N_" + std::to_string(parameters->cells_count) +
//                              "_I_" + std::to_string(parameters->knots[iter]) + ".csv");
//        whisker_box_file.write(buffer_wb.c_str(), sizeof_char*buffer_wb.size());
//        std::cout << "Whisker boxes saved: " << get_time();
//        whisker_box_file.close();
//        wbs.clear();

        iter++;
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: addresses)
        free(arr);
    for (auto arr: addresses_noisy)
        free(arr);

    return report;
}


report_map Runners::SynthRunner::kanerva_nat(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    std::string dp = data_path;
    bool* d = get_cs1(600, 12000, dp);

    int c = 0;

    std::vector<std::vector<short>> sparse_array_indices;
    std::vector<bool*> sparse_arrays;
    for (int i = 0; i < 12000; i++)
    {
        int ones = 0;
        std::vector<short> sparse_array_inds;
        bool* sparse_array = (bool*) malloc(parameters->value_length * sizeof(bool));
        for (short j = 0; j < (short) parameters->value_length; j += 1)
        {
            bool b = d[i*600 + j];
            sparse_array[j] = b;

            if (b)
            {
                sparse_array_inds.push_back(j);
                ones += 1;
            }
        }
        if (ones >= 4)
        {
            c += 1;
            sparse_array_indices.push_back(sparse_array_inds);
            sparse_arrays.push_back(sparse_array);
        }
        else {
            free(sparse_array);
        }
    }

    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length, parameters->cells_count, data_path);

    //    SDM_JAECKEL<short, short, int> sdm_jaeckel(parameters->mask_length, parameters->address_length, parameters->value_length, parameters->cells_count,
    //                                               parameters->block_count, parameters->threads_per_block,
    //                                               0.0, mask_indices=mask_indices);

    int dist = 8;
    SDM_KANERVA_SPARSE<short, short, int> sdm_kanerva(dist, parameters->mask_length, parameters->address_length,
                                                      parameters->value_length, parameters->cells_count,
                                                      parameters->block_count, parameters->threads_per_block, mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    while (all_indices.size() < parameters->max_arrays)
    {
        iter++;
        int zero_acts = 0;
        std::vector<int> indices;

        while (indices.size() != parameters->step)
        {
            bool* value = sparse_arrays[write_index];

            int acts = sdm_kanerva.write(value, value);
            if (acts >= 0)
            {
                indices.push_back(write_index);
                all_indices.push_back(write_index);
            }
            if (acts == 0)
                zero_acts += 1;
            write_index++;
        }
        std::cout << "arrays=" << iter*parameters->step << " zero_acts=" << zero_acts <<
                     " written=" << all_indices.size() << " dist=" << dist << " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/kanerva_nat/read_S_" + std::to_string(parameters->num_ones) +
        "_K_" + std::to_string(parameters->mask_length) +
        "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/kanerva_nat/read_noisy_S_" + std::to_string(parameters->num_ones) +
        "_K_" + std::to_string(parameters->mask_length) +
        "_I_" + std::to_string(iter*parameters->step) + ".csv");
        double l1 = 0.0;
        double l1_n = 0.0;
        for (auto i: all_indices)
        {
            bool* value = sparse_arrays[i];
            bool* addr = (bool*) malloc(parameters->address_length * sizeof(bool));
            for (int j = 0; j < parameters->address_length; j++)
                addr[j] = value[j];

            bool* restored = sdm_kanerva.read(addr);
            file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                l1 += (restored[j] != value[j]);
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read << restored[j] << sep;
            }

            // get noisy
            std::mt19937 generator(i);
            std::uniform_int_distribution<int> u_distribution(0, parameters->num_ones - 1);

            int swap_swap_index = u_distribution(generator);
            int swap_index = sparse_array_indices[i][swap_swap_index];

            addr[swap_index] = false;

            // read noisy
            bool* restored_noisy = sdm_kanerva.read(addr);
            file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                l1_n += (restored_noisy[j] != value[j]);
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }
            // read noisy
            free(addr);
            free(restored);
            free(restored_noisy);
        }
        std::cout << "avg_l1=" << l1/all_indices.size() << " avg_l1_n=" << l1_n/all_indices.size() << " time=" << get_time();;
        file_read.close();
        file_read_noisy.close();
    }
    for (auto sparse_array: sparse_arrays)
    {
        free(sparse_array);
    }
    free(d);

    return report;
}


report_map Runners::SynthRunner::jaeckel_nat(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    std::string dp = data_path;
    bool* d = get_cs1(600, 12000, dp);

    int c = 0;

    std::vector<std::vector<short>> sparse_array_indices;
    std::vector<bool*> sparse_arrays;
    for (int i = 0; i < 12000; i++)
    {
        int ones = 0;
        std::vector<short> sparse_array_inds;
        bool* sparse_array = (bool*) malloc(parameters->value_length * sizeof(bool));
        for (short j = 0; j < (short) parameters->value_length; j += 1)
        {
            bool b = d[i*600 + j];
            sparse_array[j] = b;

            if (b)
            {
                sparse_array_inds.push_back(j);
                ones += 1;
            }
        }
        if (ones >= 4)
        {
            c += 1;
            sparse_array_indices.push_back(sparse_array_inds);
            sparse_arrays.push_back(sparse_array);
        }
        else {
            free(sparse_array);
        }
    }

    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length, parameters->cells_count, data_path);

    //    SDM_JAECKEL<short, short, int> sdm_jaeckel(parameters->mask_length, parameters->address_length, parameters->value_length, parameters->cells_count,
    //                                               parameters->block_count, parameters->threads_per_block,
    //                                               0.0, mask_indices=mask_indices);


    SDM_LABELS<short, short, int> sdm_jaeckel(parameters->mask_length, parameters->address_length, parameters->value_length, parameters->cells_count,
                                              parameters->block_count, parameters->threads_per_block, ReadingType::STATISTICAL,
                                              0.0, mask_indices=mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    while (all_indices.size() < parameters->max_arrays)
    {
        iter++;
        int zero_acts = 0;
        std::vector<int> indices;

        while (indices.size() != parameters->step)
        {
            bool* value = sparse_arrays[write_index];

            int acts = sdm_jaeckel.write(value, value);
            if (acts >= 0)
            {
                indices.push_back(write_index);
                all_indices.push_back(write_index);
            }
            if (acts == 0)
                zero_acts += 1;
            write_index++;
        }
        std::cout << "arrays=" << iter*parameters->step << " zero_acts=" << zero_acts <<
        " written=" << all_indices.size() << " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/jaeckel_nat/read_S_" + std::to_string(parameters->num_ones) +
        "_K_" + std::to_string(parameters->mask_length) +
        "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/jaeckel_nat/read_noisy_S_" + std::to_string(parameters->num_ones) +
        "_K_" + std::to_string(parameters->mask_length) +
        "_I_" + std::to_string(iter*parameters->step) + ".csv");
        double l1 = 0.0;
        double l1_n = 0.0;
        for (auto i: all_indices)
        {
            double local_l1 = 0.0;
            bool* value = sparse_arrays[i];
            bool* addr = (bool*) malloc(parameters->address_length * sizeof(bool));
            for (int j = 0; j < parameters->address_length; j++)
                addr[j] = value[j];

            bool* restored = sdm_jaeckel.read(addr);
            file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                local_l1 += (restored[j] != value[j]);
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read << restored[j] << sep;
            }
            l1 += local_l1;

            // get noisy
            std::mt19937 generator(i);
            std::uniform_int_distribution<int> u_distribution(0, sparse_array_indices[i].size() - 1);

            int swap_swap_index = u_distribution(generator);
            int swap_index = sparse_array_indices[i][swap_swap_index];

            addr[swap_index] = false;

            // read noisy
            bool* restored_noisy = sdm_jaeckel.read(addr);
            file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                l1_n += (restored_noisy[j] != value[j]);
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }
            // read noisy
            free(addr);
            free(restored);
            free(restored_noisy);
        }
        std::cout << "avg_l1=" << l1/all_indices.size() << " avg_l1_n=" << l1_n/all_indices.size() << " time=" << get_time();;
        file_read.close();
        file_read_noisy.close();
    }
    for (auto sparse_array: sparse_arrays)
    {
        free(sparse_array);
    }
    free(d);

    return report;
}


report_map Runners::SynthRunner::cs_nat(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    std::string dp = data_path;
    bool* d = get_cs1(600, 12000, dp);

    int c = 0;

    std::vector<std::vector<short>> sparse_array_indices;
    std::vector<bool*> sparse_arrays;
    for (int i = 0; i < 12000; i++)
    {
        int ones = 0;
        std::vector<short> sparse_array_inds;
        bool* sparse_array = (bool*) malloc(parameters->address_length * sizeof(bool));
        for (short j = 0; j < (short) parameters->address_length; j += 1)
        {
            bool b = d[i*600 + j];
            sparse_array[j] = b;

            if (b)
            {
                sparse_array_inds.push_back(j);
                ones += 1;
            }
        }
        if (ones >= 4)
        {
            c += 1;
            sparse_array_indices.push_back(sparse_array_inds);
            sparse_arrays.push_back(sparse_array);
        }
        else {
            free(sparse_array);
        }
        if (sparse_arrays.size() == parameters->max_arrays)
            break;
    }

    int rows = parameters->value_length;
    int columns = parameters->address_length;

    int num_to_read = 1 * parameters->max_arrays;

    bool* array_data = (bool*) malloc(columns*num_to_read*sizeof(bool));
    bool* array_data_noisy = (bool*) malloc(columns*num_to_read*sizeof(bool));
    for (int i = 0; i < num_to_read; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            array_data[i*columns + j] = sparse_arrays[i][j];
            array_data_noisy[i*columns + j] = array_data[i*columns + j];
        }
    }

    for (int i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        const int sz = sparse_array_indices[i].size();
        std::uniform_int_distribution<int> u_distribution(0, sz - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
            );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    uint transformation_size = rows*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs_nat/matrix_" + std::to_string(parameters->value_length) + ".csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    auto transformed = (SUM_TYPE*) malloc(transformation_size*sizeof(SUM_TYPE));

    bool* cuda_data;
    cuda_malloc(&cuda_data, columns*num_to_read);
    cuda_memcpy_to_gpu(cuda_data, array_data, columns*num_to_read);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
            );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);

    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);

    std::vector<SUM_TYPE*> arrays(num_to_read);

    std::vector<bool*> addresses(num_to_read);
    std::vector<bool*> addresses_noisy(num_to_read);
    for (int i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (int j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto address = (bool*) malloc(columns*sizeof(bool));
        for (int j = 0; j < columns; j++)
        {
            auto val = array_data[i*columns + j];
            address[j] = val;
        }
        addresses[i] = address;

        auto address_noisy = (bool*) malloc(columns*sizeof(bool));
        for (int j = 0; j < columns; j++)
        {
            auto val = array_data_noisy[i*columns + j];
            address_noisy[j] = val;
        }
        addresses_noisy[i] = address_noisy;
    }

    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length,
                                                 parameters->cells_count, data_path);

    SDM_CS2<short, short, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                               parameters->value_length, parameters->cells_count, parameters->block_count,
                                               parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    while (all_indices.size() < parameters->max_arrays)
    {
        iter++;
        std::vector<int> indices;
        int zero_acts = 0;

        while (indices.size() != parameters->step)
        {
            auto value = arrays[write_index];
            auto address = addresses[write_index];

            int acts = sdm.write(value, address);
//            auto r = sdm.read(address);
//            for (int i = 0; i < 600; i++)
//                std::cout << i << "->" << address[i] << " ";
//            std::cout << std::endl;
//            for (int i = 0; i < parameters->value_length; i++)
//                std::cout << i << "->" << value[i] << " ";
//            std::cout << std::endl;
//            for (int i = 0; i < parameters->value_length; i++)
//                std::cout << i << "->" << r[i] << " ";
//            std::cout << std::endl;
            if (acts >= 0)
            {
                indices.push_back(write_index);
                all_indices.push_back(write_index);
            }
            if (acts == 0)
                zero_acts += 1;
            write_index++;
        }
        std::cout << "Finished writing |" <<
        " zero_acts=" << zero_acts <<
        " coef=" << parameters->value_length/parameters->num_ones <<
        " m=" << parameters->value_length <<
        " N=" << parameters->cells_count <<
        " K=" << parameters->mask_length <<
        " I=" << iter*parameters->step <<
        " wi=" << write_index <<
        " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_nat/read_m_" + std::to_string(parameters->value_length) +
        "_K_" + std::to_string(parameters->mask_length) +
        "_N_" + std::to_string(parameters->cells_count) +
        "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_nat/read_noisy_m_" + std::to_string(parameters->value_length) +
        "_K_" + std::to_string(parameters->mask_length) +
        "_N_" + std::to_string(parameters->cells_count) +
        "_I_" + std::to_string(iter*parameters->step) + ".csv");
        for (auto i: all_indices)
        {
            auto address = addresses[i];
            auto address_noisy = addresses_noisy[i];

            double* restored = sdm.read(address);
            file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read << restored[j] << sep;
            }

            // read noisy
            double* restored_noisy = sdm.read(address_noisy);
            file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }

            // read noisy
            free(restored);
            free(restored_noisy);
        }
        file_read.close();
        file_read_noisy.close();
        std::cout  << "Finished reading | " << " coef=" << parameters->value_length/parameters->num_ones << " m=" << parameters->value_length <<
        " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
        " I=" << iter*parameters->step << " wi=" << write_index << " time=" << get_time();
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: addresses)
        free(arr);
    for (auto arr: addresses_noisy)
        free(arr);

    return report;
}


report_map Runners::SynthRunner::cs_nat_alpha(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    std::string dp = data_path;
    bool* d = get_cs1_txt(626, 9000, dp);
//    for (int i = 0; i < 2; i++)
//    {
//        for (int j = 0; j < 626; j++)
//        {
//            std::cout << d[i*626 + j];
//        }
//        std::cout << std::endl;
//    }

    int c = 0;

    std::vector<std::vector<short>> sparse_array_indices;
    std::vector<bool*> sparse_arrays;
    for (int i = 0; i < 9000; i++)
    {
        int ones = 0;
        std::vector<short> sparse_array_inds;
        bool* sparse_array = (bool*) malloc(parameters->address_length * sizeof(bool));
        for (short j = 0; j < (short) parameters->address_length; j += 1)
        {
            bool b = d[i*parameters->address_length + j];
            sparse_array[j] = b;

            if (b)
            {
                sparse_array_inds.push_back(j);
                ones += 1;
            }
        }
        if (ones >= 4)
        {
            c += 1;
            sparse_array_indices.push_back(sparse_array_inds);
            sparse_arrays.push_back(sparse_array);
        }
        else {
            free(sparse_array);
        }
        if (sparse_arrays.size() == parameters->max_arrays)
            break;
    }

    int rows = parameters->value_length;
    int columns = parameters->address_length;

    int num_to_read = 1 * parameters->max_arrays;

    bool* array_data = (bool*) malloc(columns*num_to_read*sizeof(bool));
    bool* array_data_noisy = (bool*) malloc(columns*num_to_read*sizeof(bool));
    for (int i = 0; i < num_to_read; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            array_data[i*columns + j] = sparse_arrays[i][j];
            array_data_noisy[i*columns + j] = array_data[i*columns + j];
        }
    }

    for (int i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        const int sz = sparse_array_indices[i].size();
        std::uniform_int_distribution<int> u_distribution(0, sz - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
    );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    uint transformation_size = rows*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs_nat_alpha/matrix_" + std::to_string(parameters->value_length) + ".csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    auto transformed = (SUM_TYPE*) malloc(transformation_size*sizeof(SUM_TYPE));

    bool* cuda_data;
    cuda_malloc(&cuda_data, columns*num_to_read);
    cuda_memcpy_to_gpu(cuda_data, array_data, columns*num_to_read);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
    );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);

    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);

    std::vector<SUM_TYPE*> arrays(num_to_read);

    std::vector<bool*> addresses(num_to_read);
    std::vector<bool*> addresses_noisy(num_to_read);
    for (int i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (int j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto address = (bool*) malloc(columns*sizeof(bool));
        for (int j = 0; j < columns; j++)
        {
            auto val = array_data[i*columns + j];
            address[j] = val;
        }
        addresses[i] = address;

        auto address_noisy = (bool*) malloc(columns*sizeof(bool));
        for (int j = 0; j < columns; j++)
        {
            auto val = array_data_noisy[i*columns + j];
            address_noisy[j] = val;
        }
        addresses_noisy[i] = address_noisy;
    }

    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length,
                                                 parameters->cells_count, data_path);

    SDM_CS2<short, short, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                               parameters->value_length, parameters->cells_count, parameters->block_count,
                                               parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    std::vector<int> activations;
    while (all_indices.size() < parameters->max_arrays)
    {
        iter++;
        std::vector<int> indices;
        int zero_acts = 0;

        while (indices.size() != parameters->step)
        {
            auto value = arrays[write_index];
            auto address = addresses[write_index];

            int acts = sdm.write(value, address);
            if (acts >= 0)
            {
                indices.push_back(write_index);
                all_indices.push_back(write_index);
            }
            if (acts == 0)
                zero_acts += 1;
            activations.push_back(acts);
            write_index++;
        }
        double a = 0.0;
        for (int aaa: activations)
            a += aaa;
        a /= (double) activations.size();
        std::cout << "Finished writing |" <<
                  " zero_acts=" << zero_acts <<
                  " avg_acts=" << a <<
                  " coef=" << parameters->value_length/parameters->num_ones <<
                  " m=" << parameters->value_length <<
                  " N=" << parameters->cells_count <<
                  " K=" << parameters->mask_length <<
                  " I=" << iter*parameters->step <<
                  " wi=" << write_index <<
                  " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_nat_alpha/read_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_nat_alpha/read_noisy_m_" + std::to_string(parameters->value_length) +
                             "_K_" + std::to_string(parameters->mask_length) +
                             "_N_" + std::to_string(parameters->cells_count) +
                             "_I_" + std::to_string(iter*parameters->step) + ".csv");
        for (auto i: all_indices)
        {
            auto address = addresses[i];
            auto address_noisy = addresses_noisy[i];

            double* restored = sdm.read(address);
            file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read << restored[j] << sep;
            }

            // read noisy
            double* restored_noisy = sdm.read(address_noisy);
            file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }

            // read noisy
            free(restored);
            free(restored_noisy);
        }
        file_read.close();
        file_read_noisy.close();
        std::cout  << "Finished reading | " << " coef=" << parameters->value_length/parameters->num_ones << " m=" << parameters->value_length <<
                   " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
                   " I=" << iter*parameters->step << " wi=" << write_index << " time=" << get_time();
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: addresses)
        free(arr);
    for (auto arr: addresses_noisy)
        free(arr);

    return report;
}


report_map Runners::SynthRunner::cs_nat_balanced_impact(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    std::string dp = data_path;
    bool* d = get_cs1(600, 12000, dp);

    int c = 0;

    std::vector<std::vector<short>> sparse_array_indices;
    std::vector<bool*> sparse_arrays;
    for (int i = 0; i < 12000; i++)
    {
        int ones = 0;
        std::vector<short> sparse_array_inds;
        bool* sparse_array = (bool*) malloc(parameters->address_length * sizeof(bool));
        for (short j = 0; j < (short) parameters->address_length; j += 1)
        {
            bool b = d[i*600 + j];
            sparse_array[j] = b;

            if (b)
            {
                sparse_array_inds.push_back(j);
                ones += 1;
            }
        }
        if (ones >= 4)
        {
            c += 1;
            sparse_array_indices.push_back(sparse_array_inds);
            sparse_arrays.push_back(sparse_array);
        }
        else {
            free(sparse_array);
        }
        if (sparse_arrays.size() == parameters->max_arrays)
            break;
    }

    int rows = parameters->value_length;
    int columns = parameters->address_length;

    int num_to_read = 1 * parameters->max_arrays;

    bool* array_data = (bool*) malloc(columns*num_to_read*sizeof(bool));
    bool* array_data_noisy = (bool*) malloc(columns*num_to_read*sizeof(bool));
    for (int i = 0; i < num_to_read; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            array_data[i*columns + j] = sparse_arrays[i][j];
            array_data_noisy[i*columns + j] = array_data[i*columns + j];
        }
    }

    for (int i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        const int sz = sparse_array_indices[i].size();
        std::uniform_int_distribution<int> u_distribution(0, sz - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
            );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    uint transformation_size = rows*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs_nat_balanced_impact/matrix_" + std::to_string(parameters->value_length) + ".csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    auto transformed = (SUM_TYPE*) malloc(transformation_size*sizeof(SUM_TYPE));

    bool* cuda_data;
    cuda_malloc(&cuda_data, columns*num_to_read);
    cuda_memcpy_to_gpu(cuda_data, array_data, columns*num_to_read);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
            );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);

    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);

    std::vector<SUM_TYPE*> arrays(num_to_read);

    std::vector<bool*> addresses(num_to_read);
    std::vector<bool*> addresses_noisy(num_to_read);
    for (int i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (int j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto address = (bool*) malloc(columns*sizeof(bool));
        for (int j = 0; j < columns; j++)
        {
            auto val = array_data[i*columns + j];
            address[j] = val;
        }
        addresses[i] = address;

        auto address_noisy = (bool*) malloc(columns*sizeof(bool));
        for (int j = 0; j < columns; j++)
        {
            auto val = array_data_noisy[i*columns + j];
            address_noisy[j] = val;
        }
        addresses_noisy[i] = address_noisy;
    }

    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length,
                                                 parameters->cells_count, data_path);

    SDM_CS2<short, short, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                               parameters->value_length, parameters->cells_count, parameters->block_count,
                                               parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    while (all_indices.size() < parameters->max_arrays)
    {
        iter++;
        std::vector<int> indices;
        int zero_acts = 0;

        while (indices.size() != parameters->step)
        {
            auto value = arrays[write_index];
            auto address = addresses[write_index];

            int num = sum2<bool, int>(address, (int) parameters->address_length);
            double weight = pow(num, parameters->mask_length);
            int acts = sdm.write(value, address, weight);
            if (acts >= 0)
            {
                indices.push_back(write_index);
                all_indices.push_back(write_index);
            }
            if (acts == 0)
                zero_acts += 1;
            write_index++;
        }
        std::cout << "Finished writing |" <<
        " zero_acts=" << zero_acts <<
        " coef=" << parameters->address_length/parameters->num_ones <<
        " m=" << parameters->value_length <<
        " N=" << parameters->cells_count <<
        " K=" << parameters->mask_length <<
        " I=" << iter*parameters->step <<
        " wi=" << write_index <<
        " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_nat_balanced_impact/read_m_" + std::to_string(parameters->value_length) +
        "_K_" + std::to_string(parameters->mask_length) +
        "_N_" + std::to_string(parameters->cells_count) +
        "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_nat_balanced_impact/read_noisy_m_" + std::to_string(parameters->value_length) +
        "_K_" + std::to_string(parameters->mask_length) +
        "_N_" + std::to_string(parameters->cells_count) +
        "_I_" + std::to_string(iter*parameters->step) + ".csv");
        for (auto i: all_indices)
        {
            auto address = addresses[i];
            auto address_noisy = addresses_noisy[i];

            double* restored = sdm.read(address);
            file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read << restored[j] << sep;
            }

            // read noisy
            double* restored_noisy = sdm.read(address_noisy);
            file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }

            // read noisy
            free(restored);
            free(restored_noisy);
        }
        file_read.close();
        file_read_noisy.close();
        std::cout  << "Finished reading | " << " coef=" << parameters->value_length/parameters->num_ones << " m=" << parameters->value_length <<
        " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
        " I=" << iter*parameters->step << " wi=" << write_index << " time=" << get_time();
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: addresses)
        free(arr);
    for (auto arr: addresses_noisy)
        free(arr);

    return report;
}


report_map Runners::SynthRunner::cs3_nat(const std::string &data_path, const std::string &output_path)
{
    report_map report;
    std::string dp = data_path;
    bool* d = get_cs1(600, 12000, dp);

    int c = 0;

    std::vector<std::vector<short>> sparse_array_indices;
    std::vector<bool*> sparse_arrays;
    for (int i = 0; i < 12000; i++)
    {
        int ones = 0;
        std::vector<short> sparse_array_inds;
        bool* sparse_array = (bool*) malloc(parameters->address_length * sizeof(bool));
        for (short j = 0; j < (short) parameters->address_length; j += 1)
        {
            bool b = d[i*600 + j];
            sparse_array[j] = b;

            if (b)
            {
                sparse_array_inds.push_back(j);
                ones += 1;
            }
        }
        if (ones >= 4)
        {
            c += 1;
            sparse_array_indices.push_back(sparse_array_inds);
            sparse_arrays.push_back(sparse_array);
        }
        else {
            free(sparse_array);
        }
        if (sparse_arrays.size() == parameters->max_arrays)
            break;
    }

    int rows = parameters->value_length;
    int columns = parameters->address_length;

    int num_to_read = 1 * parameters->max_arrays;

    bool* array_data = (bool*) malloc(columns*num_to_read*sizeof(bool));
    bool* array_data_noisy = (bool*) malloc(columns*num_to_read*sizeof(bool));
    for (int i = 0; i < num_to_read; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            array_data[i*columns + j] = sparse_arrays[i][j];
            array_data_noisy[i*columns + j] = array_data[i*columns + j];
        }
    }

    for (int i = 0; i < num_to_read; i++)
    {
        // get noisy
        std::mt19937 generator(i);
        const int sz = sparse_array_indices[i].size();
        std::uniform_int_distribution<int> u_distribution(0, sz - 1);

        int swap_swap_index = u_distribution(generator);
        int swap_index = sparse_array_indices[i][swap_swap_index];

        array_data_noisy[i*columns + swap_index] = false;
    }

    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;

    cuda_malloc(&cuda_transformation, rows*columns);

    kernel_decorator(
            generate_small_random_matrix<bool>,
            parameters->block_count, parameters->threads_per_block, true,
            rows, columns, cuda_transformation
    );

    cuda_memcpy_from_gpu(transformation, cuda_transformation, rows*columns);

    uint transformation_size = rows*num_to_read;

    std::ofstream transformation_file;
    auto transformation_file_path = output_path + "/synth/cs3_nat/matrix_" + std::to_string(parameters->value_length) + ".csv";
    transformation_file.open(transformation_file_path);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            auto ind = i*columns + j;
            auto el = transformation[ind];
            int to_write = 2*el - 1;
            auto sep = (j == columns - 1) ? "\n" : ",";
            transformation_file << to_write << sep;
        }
    }
    transformation_file.close();

    typedef short SUM_TYPE;

    SUM_TYPE* cuda_transformed;

    cuda_malloc(&cuda_transformed, transformation_size);
    cuda_memset(cuda_transformed, (SUM_TYPE)0, transformation_size);

    auto transformed = (SUM_TYPE*) malloc(transformation_size*sizeof(SUM_TYPE));

    bool* cuda_data;
    cuda_malloc(&cuda_data, columns*num_to_read);
    cuda_memcpy_to_gpu(cuda_data, array_data, columns*num_to_read);

    uint thread_count = parameters->block_count*parameters->threads_per_block;
    kernel_decorator(
            mult_matrix<SUM_TYPE>,
            parameters->block_count, parameters->threads_per_block, true,
            cuda_transformation, cuda_data, cuda_transformed, rows, columns, num_to_read, thread_count
    );

    cuda_memcpy_from_gpu(transformed, cuda_transformed, transformation_size);

    cuda_free(cuda_transformation);
    cuda_free(cuda_transformed);

    std::vector<SUM_TYPE*> arrays(num_to_read);

    std::vector<bool*> addresses(num_to_read);
    std::vector<bool*> addresses_noisy(num_to_read);
    for (int i = 0; i < num_to_read; i++)
    {
        auto value = (SUM_TYPE*) malloc(rows*sizeof(SUM_TYPE));
        for (int j = 0; j < rows; j++)
        {
            auto val = transformed[j*num_to_read + i];
            value[j] = val;
        }
        arrays[i] = value;

        auto address = (bool*) malloc(columns*sizeof(bool));
        for (int j = 0; j < columns; j++)
        {
            auto val = array_data[i*columns + j];
            address[j] = val;
        }
        addresses[i] = address;

        auto address_noisy = (bool*) malloc(columns*sizeof(bool));
        for (int j = 0; j < columns; j++)
        {
            auto val = array_data_noisy[i*columns + j];
            address_noisy[j] = val;
        }
        addresses_noisy[i] = address_noisy;
    }

    auto mask_indices = read_mask_indices<short>(parameters->mask_length, parameters->address_length,
                                                 parameters->cells_count, data_path);

    SDM_CS3<short, short, short, SUM_TYPE> sdm(parameters->mask_length, parameters->address_length,
                                               parameters->value_length, parameters->cells_count, parameters->block_count,
                                               parameters->threads_per_block, mask_indices=mask_indices);

    free(mask_indices);

    int write_index = 0;
    std::vector<int> all_indices;
    int iter = 0;
    while (all_indices.size() < parameters->max_arrays)
    {
        iter++;
        std::vector<int> indices;
        int zero_acts = 0;

        while (indices.size() != parameters->step)
        {
            auto value = arrays[write_index];
            auto address = addresses[write_index];

            int acts = sdm.write(value, address);
            if (acts >= 0)
            {
                indices.push_back(write_index);
                all_indices.push_back(write_index);
            }
            if (acts == 0)
                zero_acts += 1;
            write_index++;
        }
        std::cout << "Finished writing |" <<
                  " zero_acts=" << zero_acts <<
                  " coef=" << parameters->value_length/parameters->num_ones <<
                  " m=" << parameters->value_length <<
                  " N=" << parameters->cells_count <<
                  " K=" << parameters->mask_length <<
                  " I=" << iter*parameters->step <<
                  " wi=" << write_index <<
                  " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs3_nat/read_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs3_nat/read_noisy_m_" + std::to_string(parameters->value_length) +
                             "_K_" + std::to_string(parameters->mask_length) +
                             "_N_" + std::to_string(parameters->cells_count) +
                             "_I_" + std::to_string(iter*parameters->step) + ".csv");
        for (auto i: all_indices)
        {
            auto address = addresses[i];
            auto address_noisy = addresses_noisy[i];

            double* restored = sdm.read(address);
            file_read << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read << restored[j] << sep;
            }

            // read noisy
            double* restored_noisy = sdm.read(address_noisy);
            file_read_noisy << i << "->";
            for (int j = 0; j < parameters->value_length; j++)
            {
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }

            // read noisy
            free(restored);
            free(restored_noisy);
        }
        file_read.close();
        file_read_noisy.close();
        std::cout  << "Finished reading | " << " coef=" << parameters->value_length/parameters->num_ones << " m=" << parameters->value_length <<
                   " N=" << parameters->cells_count << " K=" << parameters->mask_length <<
                   " I=" << iter*parameters->step << " wi=" << write_index << " time=" << get_time();
    }

    free(array_data);
    free(array_data_noisy);
    free(transformed);

    for (auto arr: arrays)
        free(arr);
    for (auto arr: addresses)
        free(arr);
    for (auto arr: addresses_noisy)
        free(arr);

    return report;
}
#endif //cs2_runner_cu
