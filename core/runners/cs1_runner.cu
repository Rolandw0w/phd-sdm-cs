#ifndef cs1_runner_cu
#define cs1_runner_cu


#include "cs1_runner.cuh"
#include <fstream>


report_map Runners::CS1Runner::naive(const std::string& data_path, const std::string& output_path)
{
    report_map report;

    report.insert({"mask_length", parameters->mask_length});
    report.insert({"value_length", parameters->value_length});
    report.insert({"address_length", parameters->address_length});
    report.insert({"cells_count", parameters->cells_count});
    report.insert({"image_count", parameters->image_count});
    report.insert({"images_read", parameters->images_read});
    report.insert({"block_count", parameters->block_count});
    report.insert({"threads_per_block", parameters->threads_per_block});
    report.insert({"labels_count", parameters->labels_count});
    report.insert({"target_count", parameters->target_count});
    report.insert({"bits_per_num", parameters->bits_per_num});

    int rows = parameters->target_count;
    int columns = 4*parameters->labels_count;
    bool* transformation = (bool*) malloc(rows*columns*sizeof(bool));
    bool* cuda_transformation;
    cudaMalloc((void**)&cuda_transformation, rows*columns*sizeof(bool));
    check_errors<int>("cudaMalloc/cuda_transformation");

    generate_small_random_matrix<<<parameters->block_count, parameters->threads_per_block>>>
                                (rows, columns, cuda_transformation);
    cudaMemcpy(transformation, cuda_transformation, rows*columns*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    check_errors<int>("cudaMemcpy/transformation");
    uint m = rows*parameters->image_count;

    std::ofstream transformation_file;
    transformation_file.open(output_path + "\\cs1_matrix_K_" + std::to_string(parameters->mask_length) +
                  "_I_" + std::to_string(parameters->images_read) + ".csv");

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

    typedef int SUM_TYPE;

    ulong result_bytes = m*sizeof(SUM_TYPE);

    SUM_TYPE* cuda_transformed;
    cudaMalloc((void**)&cuda_transformed, result_bytes);
    check_errors<int>("cudaMalloc/cuda_transformed");
    cudaMemset(cuda_transformed, (SUM_TYPE)0, result_bytes);
    cudaDeviceSynchronize();

    SUM_TYPE* transformed = (SUM_TYPE*) malloc(result_bytes);
    check_errors<int>("cudaMemset/cuda_transformed");

    bool* cuda_data;
    cudaMalloc((void**)&cuda_data, columns*parameters->image_count*sizeof(bool));
    check_errors<int>("cudaMalloc/cuda_data");
    cudaMemcpy(cuda_data, data, columns*parameters->image_count*sizeof(bool), cudaMemcpyHostToDevice);
    check_errors<int>("cudaMemcpy/cuda_data");

    uint tc = parameters->block_count*parameters->threads_per_block;
    cudaDeviceSynchronize();
    mult_matrix<SUM_TYPE><<<parameters->block_count, parameters->threads_per_block>>>
                (cuda_transformation, cuda_data, cuda_transformed, rows, columns, parameters->image_count, tc);
    cudaDeviceSynchronize();
    check_errors<int>("mult_matrix");
    cudaMemcpy(transformed, cuda_transformed, result_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    check_errors<int>("cudaMemcpyResult/transformed2");
    std::cout << std::endl;
    auto max = transformed[0];
    auto max_i = 0;
    auto min = transformed[0];
    auto min_i = 0;
    for (int i = 1; i < m; i++)
    {
        auto el = transformed[i];
        if (min > el)
        {
            min_i = i;
            min = el;
        }
        if (max < el)
        {
            max_i = i;
            max = el;
        }
    }
    std::cout <<std::endl;
    std::cout << "max=" << max << " max_i=" << max_i << std::endl;
    std::cout << "min=" << min << " min_i=" << min_i << std::endl << std::endl;

    std::vector<short*> images(parameters->image_count);
    long pos = 0;
    long neg = 0;
    long z = 0;
    for(int i = 0; i < parameters->image_count; i++)
    {
        auto image = (short*) malloc((rows * sizeof(short)));
        for(int j = 0; j < rows; j++)
        {
            auto el = transformed[j*parameters->image_count+i];
            image[j] = el;
            if (el > 0)
                pos++;
            if (el < 0)
                neg++;
            if (el == 0)
                z++;
        }
        images[i] = image;
    }

    cudaFree(cuda_transformation);
    cudaFree(cuda_transformed);

    SDM_CS1<int, short, short, short> sdm(parameters->mask_length, parameters->address_length,
                                            parameters->value_length, parameters->cells_count, parameters->block_count,
                                            parameters->threads_per_block);

    long write_time_start = clock();
    std::cout << "Started writing ";
    int act_zero = 0;
    std::vector<int> acts(parameters->image_count);
    for(int i = 0; i < parameters->images_read; i++)
    {
        short* image = images[i];
        int act = sdm.write(image);
        act_zero += (act == 0);
        acts[i] = act;
        if ((i+1) % 1000 == 0)
            std::cout << (i+1) << " ";
    }
    std::cout << std::endl ;
    long write_time = clock() - write_time_start;

    double sum_l1 = 0;
    long read_time_start = clock();
    std::cout << "Started reading" << " ";
    double sum_l1_arr = 0;

    double max_l1 = 0;
    double max_l1_ind = -1;

    double min_l1 = 1e12;
    double min_l1_ind = -1;

    double avg_l1_r = 0;
    double avg_l1_f = 0;
    double avg_l1_c = 0;
    int read_zeros = 0;
    std::ofstream restored;
    restored.open(output_path + "\\cs1_K_" + std::to_string(parameters->mask_length) +
                    "_I_" + std::to_string(parameters->images_read) + ".csv");
    for(int i = 0; i < parameters->images_read; i++)
    {
        auto el = images[i];
        double l1_arr = 0;
        double* remembered = sdm.read(el);
        double l1 = 0;
        double l1_r = 0;
        double l1_f = 0;
        double l1_c = 0;
        std::vector<double> rem_arr(parameters->value_length);
        std::vector<short> img_arr(parameters->value_length);
        bool is_zeros = true;
        for (int j = 0; j < parameters->value_length; j++)
        {
            double rem = remembered[j];
            auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
            restored << rem << sep;
            if (abs(rem) > 1e-6)
                is_zeros = false;
            rem_arr[j] = rem;
            auto elj = el[j];
            img_arr[j] = elj;
            double elj_r = round(rem);
            double elj_f = floor(rem);
            double elj_c = ceil(rem);
            l1_r += abs(elj_r - elj);
            l1_f += abs(elj_f - elj);
            l1_c += abs(elj_c - elj);
            l1 += abs(rem - elj);
        }
        if (is_zeros)
        {
            read_zeros += 1;
            continue;
        }
        if (max_l1 < l1)
        {
            max_l1 = l1;
            max_l1_ind = i;
        }
        if (min_l1 > l1)
        {
            min_l1 = l1;
            min_l1_ind = i;
        }

        sum_l1 += l1;
        l1_arr += l1;
        sum_l1_arr += l1_arr;
        avg_l1_r += l1_r / parameters->images_read;
        avg_l1_f += l1_f / parameters->images_read;
        avg_l1_c += l1_c / parameters->images_read;
        free(remembered);
        if ((i+1) % 1000 == 0)
            std::cout << (i+1) << " ";
    }
    std::cout << std::endl;
    long read_time = clock() - read_time_start;
    restored.close();

    report.insert({"act_zero", act_zero});
    report.insert({"avg_l1", sum_l1 / parameters->images_read});
    report.insert({"avg_l1_r", avg_l1_r});
    report.insert({"avg_l1_f", avg_l1_f});
    report.insert({"avg_l1_c", avg_l1_c});
    report.insert({"max_l1", max_l1});
    report.insert({"max_l1_ind", max_l1_ind});
    report.insert({"min_l1", min_l1});
    report.insert({"min_l1_ind", min_l1_ind});
    report.insert({"mae", sum_l1_arr / parameters->images_read / rows});
    report.insert({"avg_read_time", (double)read_time / parameters->images_read});
    report.insert({"avg_write_time", (double)write_time / parameters->images_read});
    report.insert({"read_zeros", read_zeros});
//    report.insert({"min_activations", sdm.get_min_activations()});
//    report.insert({"max_activations", sdm.get_max_activations()});
//    report.insert({"activated_cells_count", sdm.get_activations_num()});

    return report;
}

#endif //cs1_runner_cu
