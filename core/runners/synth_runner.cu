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
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/jaeckel/read_S_" + std::to_string(parameters->num_ones) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/jaeckel/read_noisy_S_" + std::to_string(parameters->num_ones) +
                             "_K_" + std::to_string(parameters->mask_length) +
                             "_I_" + std::to_string(iter*parameters->step) + ".csv");
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
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                file_read_noisy << restored_noisy[j] << sep;
            }

            // read noisy
            free(addr);
            free(restored);
            free(restored_noisy);
        }
        file_read.close();
        file_read_noisy.close();
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

    auto mask_indices = read_mask_indices_s<short>(parameters->mask_length, parameters->address_length,
                                                   parameters->cells_count, data_path, parameters->num_ones);

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
        std::cout << "Finished writing |" <<
                  " s=" << parameters->num_ones <<
                  " coef=" << parameters->value_length/parameters->num_ones <<
                  " m=" << parameters->value_length <<
                  " N=" << parameters->cells_count <<
                  " K=" << parameters->mask_length <<
                  " I=" << iter*parameters->step <<
                  " wi=" << write_index <<
                  " time=" << get_time();

        std::ofstream file_read;
        file_read.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s" + std::to_string(parameters->num_ones) +
                       "/read_m_" + std::to_string(parameters->value_length) +
                       "_K_" + std::to_string(parameters->mask_length) +
                       "_N_" + std::to_string(parameters->cells_count) +
                       "_I_" + std::to_string(iter*parameters->step) + ".csv");

        std::ofstream file_read_noisy;
        file_read_noisy.open("/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s" + std::to_string(parameters->num_ones) +
                             "/read_noisy_m_" + std::to_string(parameters->value_length) +
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
        std::cout  << "Finished reading | " <<
                   "s=" << parameters->num_ones << " coef=" << parameters->value_length/parameters->num_ones << " m=" << parameters->value_length <<
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
