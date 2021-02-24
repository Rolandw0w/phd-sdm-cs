#ifndef cs1_runner_cu
#define cs1_runner_cu


#include "cs1_runner.cuh"


report_map Runners::CS1Runner::naive(const double confidence, const bool save_images, const std::string &data_path)
{
    report_map report;

    report.insert({"mask_length", parameters->mask_length});
    report.insert({"value_length", parameters->value_length});
    report.insert({"address_length", parameters->address_length});
    report.insert({"cells_count", parameters->cells_count});
    report.insert({"image_count", parameters->image_count});
    report.insert({"block_count", parameters->block_count});
    report.insert({"threads_per_block", parameters->threads_per_block});
    report.insert({"labels_count", parameters->labels_count});
    report.insert({"target_count", parameters->target_count});
    report.insert({"bits_per_num", parameters->bits_per_num});

    int rows = parameters->target_count;
    int columns = parameters->labels_count;
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
    for (int i = 1; i < m; i++)
    {
        auto el = transformed[i];
        if (max < el)
        {
            max_i = i;
            max = el;
        }
    }
    std::cout << "max=" << max << " max_i=" << max_i << std::endl << std::endl;

    std::vector<std::vector<int>> images(parameters->image_count);
    std::vector<bool*> images_bits(parameters->image_count);
    for(int i = 0; i < parameters->image_count; i++)
    {
        std::vector<int> image(rows);
        for(int j = 0; j < rows; j++)
        {
            auto el = transformed[i+j*9000];
            image[j] = el;
        }
        images[i] = image;
    }

    uint dim = parameters->bits_per_num;
    for(int i = 0; i < parameters->image_count; i++)
    {
        auto image = images[i];
        bool* image_bits = (bool*) malloc(rows*dim*sizeof(bool));
        for(int j = 0; j < rows; j++)
        {
            auto el = image[j];
            //std::cout << el << ",,,|";
            bool* bits = to_bits(el, dim);
            for(int k = 0; k < dim; k++)
            {
                image_bits[j*dim + k] = bits[k];
            }
            free(bits);
        }
        images_bits[i] = image_bits;
    }

//    for(int i = 100; i < 101; i++)
//    {
//        auto ib = images_bits[i];
//        std::cout << std::endl;
//        for (int j = 0; j < rows; j++)
//        {
//            for (int k = 0; k < dim; k++)
//            {
//                std::cout << ib[j*dim + k];
//            }
//            std::cout << "|";
//        }
//        std::cout << std::endl;
//    }

    SDM_JAECKEL<short, short, short> sdm(parameters->mask_length, parameters->address_length,
                                         parameters->value_length, parameters->cells_count, parameters->block_count,
                                         parameters->threads_per_block);

    long write_time_start = clock();
    std::cout << "Started writing" << std::endl;
    for(int i = 0; i < parameters->image_count; i++)
    {
        auto el = images_bits[i];
        sdm.write(el);
    }
    long write_time = clock() - write_time_start;

    double sum_dist = 0;
    double sum_l1 = 0;
    long read_time_start = clock();
    //sdm.print_state();
    std::cout << "Started reading" << std::endl;
    double sum_l1_arr = 0;
    for(int i = 0; i < parameters->image_count; i++)
    {
        auto el = images_bits[i];
//        for(int j = 0; j < parameters->value_length; j++)
//        {
//            std::cout << el[j] << "|";
//        }
//        std::cout << std::endl;
        double l1_arr = 0;
        bool* remembered = sdm.read(el);
        int dist = hamming_distance(el, remembered, parameters->value_length);
        sum_dist += dist;
        for(int j = 0; j < parameters->value_length / dim; j++)
        {
            int remembered_int = 0;
            int el_int = 0;
            for(int k = 0; k < dim; k++)
            {
                bool rem = remembered[j*dim + k];
                bool el_b = el[j*dim + k];
                //std::cout << rem << "|" << el_b << ",";
                remembered_int += rem ? pow(2, dim - k - 1) : 0;
                el_int += el_b ? pow(2, dim - k - 1) : 0;
            }
            //std::cout << remembered_int << ":" << el_int << ";";
            int l1 = abs(remembered_int - el_int);
            l1_arr += l1;
            sum_l1 += l1;
        }
        sum_l1_arr += l1_arr;
        //std::cout << dist << "|";
//        for(int j = 0; j < parameters->value_length; j++)
//        {
//            std::cout << remembered[j] << "|";
//        }
//        std::cout << std::endl;
//        std::cout << dist << "|";
        free(remembered);
    }
    long read_time = clock() - read_time_start;

    report.insert({"avg_dist", sum_dist / parameters->image_count});
    report.insert({"avg_l1", sum_l1 / parameters->image_count});
    report.insert({"mae", sum_l1_arr / parameters->image_count / rows});
    report.insert({"avg_read_time", (double)read_time / parameters->image_count});
    report.insert({"avg_write_time", (double)write_time / parameters->image_count});
    report.insert({"min_activations", sdm.get_min_activations()});
    report.insert({"max_activations", sdm.get_max_activations()});
    report.insert({"activated_cells_count", sdm.get_activations_num()});

    return report;
}

#endif //cs1_runner_cu
