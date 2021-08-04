#ifndef main_cu
#define main_cu

#include <functional>
#include <iostream>
#include <map>

#include "main.cuh"


std::string reports_root;
std::string data_root;
std::string output_root;


Runners::CIFAR10RunnerParameters* get_cifar10_parameters()
{
    const uint image_count = 24000;
    const uint block_count = 64;
    const uint threads_per_block = 512;
    const uint mask_length = 14;
    const uint cells_count = 70000;
    const uint address_length = 24576;
    const uint value_length = 24576;
    static Runners::CIFAR10RunnerParameters cifar10_parameters(image_count, block_count, threads_per_block,
                                                               mask_length, cells_count, address_length, value_length);

    return &cifar10_parameters;
}

Runners::CS1RunnerParameters* get_cs1_parameters()
{
    const uint image_count = 9000;
    const uint image_read = 9000;
    const uint block_count = 64;
    const uint threads_per_block = 1024;
    const uint bits_per_num = 1;
    const uint mask_length = 10;
    const uint labels_count = 150;
    const uint target_count = 150;
    const uint address_length = bits_per_num*target_count;
    const uint value_length = bits_per_num*target_count;
    const uint cells_count = 12*1000*1000;
    auto* cs1_parameters = new Runners::CS1RunnerParameters(image_count, image_read, block_count, threads_per_block,
                                                            mask_length, cells_count, address_length, value_length,
                                                            labels_count, target_count, bits_per_num);

    return cs1_parameters;
}

Runners::CS2RunnerParameters* get_cs2_parameters()
{
    const uint image_count = 9000;
    const uint image_read = 9000;
    const uint block_count = 64;
    const uint threads_per_block = 1024;
    const uint bits_per_num = 1;
    const uint mask_length = 10;
    const uint labels_count = 150;
    const uint target_count = 150;
    const uint address_length = 600;
    const uint value_length = bits_per_num*target_count;
    const uint cells_count = 12*1000*1000;
    const uint min_features = 3;
    auto* cs2_parameters = new Runners::CS2RunnerParameters(image_count, image_read, block_count, threads_per_block,
                                                            mask_length, cells_count, address_length, value_length,
                                                            labels_count, target_count, bits_per_num, min_features);

    return cs2_parameters;
}

Runners::CS2S2RunnerParameters* get_cs2_s2_parameters()
{
    const uint image_count = 9000;
    const uint image_read = 9000;
    const uint block_count = 64;
    const uint threads_per_block = 1024;
    const uint bits_per_num = 1;
    const uint mask_length = 10;
    const uint labels_count = 150;
    const uint target_count = 150;
    const uint address_length = 600;
    const uint value_length = bits_per_num*target_count;
    const uint cells_count = 12*1000*1000;
    const uint min_features = 3;
    auto* cs2_parameters = new Runners::CS2S2RunnerParameters(image_count, image_read, block_count, threads_per_block,
                                                              mask_length, cells_count, address_length, value_length,
                                                              labels_count, target_count, bits_per_num, min_features);

    return cs2_parameters;
}

Runners::LabelsRunnerParameters* get_labels_parameters(ReadingType reading_type, double bio_threshold)
{
    const uint image_count = 9000;
    const uint image_read = 9000;
    const uint block_count = 32;
    const uint threads_per_block = 512;
    const uint mask_length = 2;
    const uint address_length = 600;
    const uint value_length = 600;
    const uint labels_count = 600;
    const uint cells_count = 3*1000*1000;
    auto* labels_parameters = new Runners::LabelsRunnerParameters(image_count, image_read, block_count, threads_per_block,
                                                                  mask_length, cells_count, address_length, value_length,
                                                                  labels_count, reading_type, bio_threshold);

    return labels_parameters;
}

Runners::KanervaRunnerParameters* get_kanerva_parameters()
{
    const uint image_count = 9000;
    const uint image_read = 9000;
    const uint block_count = 64;
    const uint threads_per_block = 1024;
    const uint max_dist = 1;
    const uint address_length = 600;
    const uint value_length = 600;
    const uint cells_count = 3000*1000;
    const double p0 = 0.95;
    auto* labels_parameters = new Runners::KanervaRunnerParameters(image_count, image_read, block_count, threads_per_block,
                                                                   max_dist, cells_count, address_length, value_length, p0);

    return labels_parameters;
}

Runners::SynthRunnerParameters* get_synth_parameters()
{
    const uint image_count = 9000;
    const uint image_read = 9000;
    const uint block_count = 64;
    const uint threads_per_block = 1024;
    const uint mask_length = 3;
    const uint address_length = 600;
    const uint value_length = 600;
    const uint cells_count = 8*1000*1000;
    const uint s = 4;
    const uint step = 500;
    const uint max_arrays = 25 * 1000;

    auto* synth_parameters = new Runners::SynthRunnerParameters(image_count, image_read, block_count, threads_per_block,
                                                                mask_length, cells_count, address_length, value_length,
                                                                s, step, max_arrays);

    return synth_parameters;
}

void print_report(report_map* report)
{
    for (const auto& elem : *report)
    {
        std::cout << elem.first << "=" << elem.second << std::endl;
    }
    std::cout << std::endl;
}

void save_report_vector_json(std::vector<report_map>* reports, std::ofstream& file)
{
    file << "[" << std::endl;
    for (const auto& report : *reports)
    {
        file << "    {" << std::endl;
        for (const auto& elem : report)
        {
            file << "        \"" << elem.first << "\"" << ": " << elem.second << "," << std::endl;
        }
        file << "    }," << std::endl;
    }
    file << "]" << std::endl;
}

void cifar10_naive()
{
    const double confidence = 0.95;
    const uint min_mask_length = 8;
    const uint max_mask_length = 32;
    const uint mask_length_step = 1;

    Runners::CIFAR10RunnerParameters* cifar10_parameters = get_cifar10_parameters();
    bool** data = get_cifar10_images(cifar10_parameters->image_count, data_root);

    std::vector<report_map> reports;
    for (uint mask_length = min_mask_length; mask_length <= max_mask_length; mask_length += mask_length_step)
    {
        Runners::CIFAR10Runner cifar10_runner{};
        cifar10_parameters->mask_length = mask_length;
        cifar10_runner.set_parameters(cifar10_parameters);
        cifar10_runner.set_data(&data);

        report_map naive_report = cifar10_runner.naive(confidence);
        reports.push_back(naive_report);
        print_report(&naive_report);
    }
    std::ofstream naive;
    naive.open(reports_root + "/naive.txt");
    save_report_vector_json(&reports, naive);

    naive.close();
    free(data);
}

void cifar10_multiple_write()
{
    const uint error_bits = 500;
    const uint min_write_count = 5;
    const uint max_write_count = 50;
    const uint write_count_step = 5;

    Runners::CIFAR10RunnerParameters* cifar10_parameters = get_cifar10_parameters();
    Runners::CIFAR10Runner cifar10_runner{};
    cifar10_runner.set_parameters(cifar10_parameters);

    bool** data = get_cifar10_images(cifar10_parameters->image_count, data_root);
    cifar10_runner.set_data(&data);

    std::vector<report_map> reports;

    report_map multiple_write_report = cifar10_runner.multiple_write(error_bits, 1);
    reports.push_back(multiple_write_report);
    for (uint write_count = min_write_count; write_count <= max_write_count; write_count += write_count_step)
    {
        multiple_write_report = cifar10_runner.multiple_write(error_bits, write_count);
        reports.push_back(multiple_write_report);
        print_report(&multiple_write_report);
    }
    std::ofstream multiple_write;
    multiple_write.open(reports_root + "/cifar10_multiple_write.txt");
    save_report_vector_json(&reports, multiple_write);

    multiple_write.close();
    free(data);
}

void cifar10_iterative_read()
{
    const uint error_bits = 500;
    const uint min_iterations_count = 1;
    const uint max_iterations_count = 25;
    const uint iterations_count_step = 1;

    Runners::CIFAR10RunnerParameters* cifar10_parameters = get_cifar10_parameters();

    Runners::CIFAR10Runner cifar10_runner{};
    cifar10_runner.set_parameters(cifar10_parameters);

    bool** data = get_cifar10_images(cifar10_parameters->image_count, data_root);
    cifar10_runner.set_data(&data);

    std::vector<report_map> reports;
    for (uint iterations_count = min_iterations_count; iterations_count <= max_iterations_count; iterations_count += iterations_count_step)
    {
        report_map iterative_read_report = cifar10_runner.iterative_read(iterations_count, error_bits);
        reports.push_back(iterative_read_report);
        print_report(&iterative_read_report);
    }
    std::ofstream iterative_read;
    iterative_read.open(reports_root + "/iterative_read.txt");
    save_report_vector_json(&reports, iterative_read);

    iterative_read.close();
    free(data);
}

void cifar10_noisy_address()
{
    const uint min_error_bits = 50;
    const uint max_error_bits = 500;
    const uint error_bits_step = 25;

    Runners::CIFAR10RunnerParameters* cifar10_parameters = get_cifar10_parameters();

    Runners::CIFAR10Runner cifar10_runner{};
    cifar10_runner.set_parameters(cifar10_parameters);

    bool** data = get_cifar10_images(cifar10_parameters->image_count, data_root);
    cifar10_runner.set_data(&data);

    std::vector<report_map> reports;
    for (uint error_bits = min_error_bits; error_bits <= max_error_bits; error_bits += error_bits_step)
    {
        report_map noisy_address_report = cifar10_runner.noisy_address(error_bits);
        reports.push_back(noisy_address_report);
        print_report(&noisy_address_report);
    }
    std::ofstream noisy_address;
    noisy_address.open(reports_root + "/noisy_address.txt");
    save_report_vector_json(&reports, noisy_address);

    noisy_address.close();
    free(data);
}

void cifar10_noisy_address_noisy_value()
{
    const uint min_error_bits = 50;
    const uint max_error_bits = 500;
    const uint error_bits_step = 25;

    Runners::CIFAR10RunnerParameters* cifar10_parameters = get_cifar10_parameters();

    Runners::CIFAR10Runner cifar10_runner{};
    cifar10_runner.set_parameters(cifar10_parameters);

    bool** data = get_cifar10_images(cifar10_parameters->image_count, data_root);
    cifar10_runner.set_data(&data);

    std::vector<report_map> reports;
    for (uint error_bits = min_error_bits; error_bits <= max_error_bits; error_bits += error_bits_step)
    {
        report_map noisy_address_report = cifar10_runner.noisy_address_noisy_value(error_bits);
        reports.push_back(noisy_address_report);
        print_report(&noisy_address_report);
    }
    std::ofstream noisy_address_noisy_value;
    noisy_address_noisy_value.open(reports_root + "/noisy_address_noisy_value.txt");
    save_report_vector_json(&reports, noisy_address_noisy_value);

    noisy_address_noisy_value.close();
    free(data);
}

void labels_stat_naive()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool** data = get_labels(labels_count, image_num, data_root);

    const uint min_mask_length = 2;
    const uint max_mask_length = 2;
    const uint mask_length_step = 1;

    Runners::LabelsRunnerParameters* labels_parameters = get_labels_parameters(ReadingType::STATISTICAL);

    std::vector<report_map> reports;
    for (uint mask_length = min_mask_length; mask_length <= max_mask_length; mask_length += mask_length_step)
    {
        Runners::LabelsRunner labels_runner{};
        labels_parameters->mask_length = mask_length;
        labels_runner.set_parameters(labels_parameters);
        labels_runner.set_data(&data);

        report_map naive_report = labels_runner.naive(data_root, output_root);
        reports.push_back(naive_report);
        print_report(&naive_report);
    }
    std::ofstream labels_stat_naive;
    labels_stat_naive.open(reports_root + "/labels_stat_naive.txt");
    save_report_vector_json(&reports, labels_stat_naive);

    labels_stat_naive.close();
    free(data);
    delete(labels_parameters);
}

void labels_knots()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool** data = get_labels(labels_count, image_num, data_root);

    uint images_reads[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};
    uint mask_lengths[] = {1, 2, 3, 4, 5, 6};
    Runners::LabelsRunnerParameters* labels_parameters = get_labels_parameters(ReadingType::STATISTICAL);

    std::vector<report_map> reports;
    for (uint mask_length: mask_lengths)
    {
        for (uint images_read: images_reads)
        {
            Runners::LabelsRunner labels_runner{};
            labels_parameters->images_read = images_read;
            labels_parameters->mask_length = mask_length;

            labels_runner.set_parameters(labels_parameters);
            labels_runner.set_data(&data);

            report_map naive_report = labels_runner.naive(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }
    std::ofstream labels_stat_naive;
    labels_stat_naive.open(reports_root + "/labels_knots.txt");
    save_report_vector_json(&reports, labels_stat_naive);

    labels_stat_naive.close();
    free(data);
    delete(labels_parameters);
}

void labels_bio_naive()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool** data = get_labels(labels_count, image_num, data_root);

    const double bio_thresholds[7] = {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99};

    Runners::LabelsRunnerParameters* labels_parameters = get_labels_parameters(ReadingType::BIOLOGICAL);

    std::vector<report_map> reports;
    for (double bio_threshold : bio_thresholds)
    {
        labels_parameters->bio_threshold = bio_threshold;

        Runners::LabelsRunner labels_runner{};
        labels_runner.set_parameters(labels_parameters);
        labels_runner.set_data(&data);

        report_map naive_report = labels_runner.naive(data_root, output_root);
        reports.push_back(naive_report);
        print_report(&naive_report);
    }

    std::ofstream labels_bio_naive;
    labels_bio_naive.open(reports_root + "/labels_bio_naive.txt");
    save_report_vector_json(&reports, labels_bio_naive);

    labels_bio_naive.close();
    free(data);
    delete(labels_parameters);
}

void cs1_naive_grid1()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS1RunnerParameters* cs1_parameters = get_cs1_parameters();

    const uint min_mask_length = 8;
    const uint max_mask_length = 32;
    const uint mask_length_step = 1;

    uint cells_counts[] = {50*1000, 100*1000, 250*1000, 500*1000, 750*1000, 1000*1000, 1250*1000, 1500*1000, 2000*1000};

    std::vector<report_map> reports;
    for (auto cells_count: cells_counts)
    {
        for (uint mask_length = min_mask_length; mask_length <= max_mask_length; mask_length += mask_length_step)
        {
            Runners::CS1Runner cs1_runner{};

            cs1_parameters->mask_length = mask_length;
            cs1_parameters->cells_count = cells_count;

            cs1_runner.set_parameters(cs1_parameters);
            cs1_runner.set_data(&data);

            report_map naive_report = cs1_runner.naive(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs1_naive;
    cs1_naive.open(reports_root + "/cs1_naive_grid1.txt");
    save_report_vector_json(&reports, cs1_naive);

    cs1_naive.close();
    free(data);
    delete(cs1_parameters);
}

void cs1_naive_grid2()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS1RunnerParameters* cs1_parameters = get_cs1_parameters();

    const uint min_mask_length = 1;
    const uint max_mask_length = 10;
    const uint mask_length_step = 1;

    uint cells_counts[] = {//50*1000, 100*1000, 250*1000, 500*1000, 750*1000, 1000*1000, 1250*1000, 1500*1000,
                           12*1000*1000};

    std::vector<report_map> reports;
    for (auto cells_count: cells_counts)
    {
        for (uint mask_length = max_mask_length; mask_length >= min_mask_length; mask_length -= mask_length_step)
        {
            Runners::CS1Runner cs1_runner{};

            cs1_parameters->mask_length = mask_length;
            cs1_parameters->cells_count = cells_count;

            cs1_runner.set_parameters(cs1_parameters);
            cs1_runner.set_data(&data);

            report_map naive_report = cs1_runner.naive(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs1_naive;
    cs1_naive.open(reports_root + "/cs1_naive_grid2.txt");
    save_report_vector_json(&reports, cs1_naive);

    cs1_naive.close();
    free(data);
    delete(cs1_parameters);
}

void kanerva_run()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool** data = get_labels(labels_count, image_num, data_root);

    Runners::KanervaRunnerParameters* kanerva_parameters = get_kanerva_parameters();
    uint max_dists[] = {1, 2, 3, 4, 5, 6};
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
                           5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};
    double p0s[] = {0.99, 0.995};

    std::vector<report_map> reports;
    for (auto p0: p0s)
    {
        for (auto max_dist: max_dists)
        {
            if (p0 == 0.99 && max_dist <= 6)
                continue;
            Runners::KanervaRunner kanerva_runner{};

            // kanerva_parameters->images_read = images_read;
            kanerva_parameters->max_dist = max_dist;
            kanerva_parameters->p0 = p0;

            kanerva_runner.set_parameters(kanerva_parameters);
            kanerva_runner.set_data(&data);

            report_map naive_report = kanerva_runner.naive(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs1_naive;
    cs1_naive.open(reports_root + "/kanerva_naive_grid2.txt");
    save_report_vector_json(&reports, cs1_naive);

    cs1_naive.close();
    free(data);
    delete(kanerva_parameters);
}

void cs1_image_count_grid()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS1RunnerParameters* cs1_parameters = get_cs1_parameters();
    uint mask_lengths[] = {8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto images_read: image_counts)
        {
            Runners::CS1Runner cs1_runner{};

            cs1_parameters->images_read = images_read;
            cs1_parameters->mask_length = mask_length;

            cs1_runner.set_parameters(cs1_parameters);
            cs1_runner.set_data(&data);

            report_map naive_report = cs1_runner.naive(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs1_naive;
    cs1_naive.open(reports_root + "/cs1_naive_grid2.txt");
    save_report_vector_json(&reports, cs1_naive);

    cs1_naive.close();
    free(data);
    delete(cs1_parameters);
}

void cs1_noisy()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS1RunnerParameters* cs1_parameters = get_cs1_parameters();
    uint mask_lengths[] = {8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};
//    uint mask_lengths[] = {15};
//    uint image_counts[] = {500};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto images_read: image_counts)
        {
            Runners::CS1Runner cs1_runner{};

            cs1_parameters->images_read = images_read;
            cs1_parameters->mask_length = mask_length;

            cs1_runner.set_parameters(cs1_parameters);
            cs1_runner.set_data(&data);

            report_map naive_report = cs1_runner.noisy(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs1_naive;
    cs1_naive.open(reports_root + "/cs1_noisy_1_grid.txt");
    save_report_vector_json(&reports, cs1_naive);

    cs1_naive.close();
    free(data);
    delete(cs1_parameters);
}

void cs1_noisy_2()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS1RunnerParameters* cs1_parameters = get_cs1_parameters();
    uint mask_lengths[] = {8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};
//    uint mask_lengths[] = {15};
//    uint image_counts[] = {500};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto images_read: image_counts)
        {
            Runners::CS1Runner cs1_runner{};

            cs1_parameters->images_read = images_read;
            cs1_parameters->mask_length = mask_length;

            cs1_runner.set_parameters(cs1_parameters);
            cs1_runner.set_data(&data);

            report_map naive_report = cs1_runner.noisy_2(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs1_naive;
    cs1_naive.open(reports_root + "/cs1_noisy_2_grid.txt");
    save_report_vector_json(&reports, cs1_naive);

    cs1_naive.close();
    free(data);
    delete(cs1_parameters);
}

void cs2_naive()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS2RunnerParameters* cs2_parameters = get_cs2_parameters();
    uint mask_lengths[] = {1, 2, 3, 4, 5};
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto images_read: image_counts)
        {
            Runners::CS2Runner cs2_runner{};

            cs2_parameters->images_read = images_read;
            cs2_parameters->mask_length = mask_length;

            cs2_runner.set_parameters(cs2_parameters);
            cs2_runner.set_data(&data);

            report_map naive_report = cs2_runner.naive(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs2_naive;
    cs2_naive.open(reports_root + "/cs2_naive.txt");
    save_report_vector_json(&reports, cs2_naive);

    cs2_naive.close();
    free(data);
    delete(cs2_parameters);
}

void cs2_noisy_1()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS2RunnerParameters* cs2_parameters = get_cs2_parameters();
    uint mask_lengths[] = {3};
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto images_read: image_counts)
        {
            Runners::CS2Runner cs2_runner{};

            cs2_parameters->images_read = images_read;
            cs2_parameters->mask_length = mask_length;

            cs2_runner.set_parameters(cs2_parameters);
            cs2_runner.set_data(&data);

            report_map naive_report = cs2_runner.noisy(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs2_naive;
    cs2_naive.open(reports_root + "/cs2_naive.txt");
    save_report_vector_json(&reports, cs2_naive);

    cs2_naive.close();
    free(data);
    delete(cs2_parameters);
}

void cs2_noisy_2()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS2RunnerParameters* cs2_parameters = get_cs2_parameters();
    uint mask_lengths[] = {3};
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto images_read: image_counts)
        {
            Runners::CS2Runner cs2_runner{};

            cs2_parameters->images_read = images_read;
            cs2_parameters->mask_length = mask_length;

            cs2_runner.set_parameters(cs2_parameters);
            cs2_runner.set_data(&data);

            report_map naive_report = cs2_runner.noisy_2(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs2_naive;
    cs2_naive.open(reports_root + "/cs2_naive.txt");
    save_report_vector_json(&reports, cs2_naive);

    cs2_naive.close();
    free(data);
    delete(cs2_parameters);
}

void cs2_naive_geq_3()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS2RunnerParameters* cs2_parameters = get_cs2_parameters();
    uint mask_lengths[] = {3};
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto images_read: image_counts)
        {
            Runners::CS2Runner cs2_runner{};

            cs2_parameters->images_read = images_read;
            cs2_parameters->mask_length = mask_length;

            cs2_runner.set_parameters(cs2_parameters);
            cs2_runner.set_data(&data);

            report_map naive_report = cs2_runner.naive_geq_3(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs2_naive;
    cs2_naive.open(reports_root + "/cs2_geq_4_naive.txt");
    save_report_vector_json(&reports, cs2_naive);

    cs2_naive.close();
    free(data);
    delete(cs2_parameters);
}

void cs2_naive_geq_3_s1()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS2RunnerParameters* cs2_parameters = get_cs2_parameters();
    uint mask_lengths[] = {3, 4, 5};
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto images_read: image_counts)
        {
            Runners::CS2Runner cs2_runner{};

            cs2_parameters->images_read = images_read;
            cs2_parameters->mask_length = mask_length;

            cs2_runner.set_parameters(cs2_parameters);
            cs2_runner.set_data(&data);

            report_map naive_report = cs2_runner.naive_geq_3_s1(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs2_naive;
    cs2_naive.open(reports_root + "/cs2_geq_3_s1.txt");
    save_report_vector_json(&reports, cs2_naive);

    cs2_naive.close();
    free(data);
    delete(cs2_parameters);
}

void cs2_s2_naive_geq_3()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    Runners::CS2S2RunnerParameters* cs2_s2_parameters = get_cs2_s2_parameters();
    uint mask_lengths[] = {8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto images_read: image_counts)
        {
            Runners::CS2S2Runner cs2_s1_runner{};

            cs2_s2_parameters->images_read = images_read;
            cs2_s2_parameters->mask_length = mask_length;

            cs2_s1_runner.set_parameters(cs2_s2_parameters);
            cs2_s1_runner.set_data(&data);

            report_map naive_report = cs2_s1_runner.naive_geq_3(data_root, output_root);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs2_naive;
    cs2_naive.open(reports_root + "/cs2_geq_3_s1.txt");
    save_report_vector_json(&reports, cs2_naive);

    cs2_naive.close();
    free(data);
    delete(cs2_s2_parameters);
}

void synth_jaeckel()
{
    Runners::SynthRunnerParameters* synth_parameters = get_synth_parameters();
    uint num_ones_arr[] = {4, 5, 6, 7, 8, 9, 10};

    std::vector<report_map> reports;
    for (auto num_ones: num_ones_arr)
    {
        Runners::SynthRunner synth_runner{};

        synth_parameters->num_ones = num_ones;

        synth_runner.set_parameters(synth_parameters);

        report_map jaeckel_report = synth_runner.jaeckel(data_root, output_root);
        reports.push_back(jaeckel_report);
        print_report(&jaeckel_report);
    }

    std::ofstream synth_jaeckel_file;
    synth_jaeckel_file.open(reports_root + "/synth_jaeckel.txt");
    save_report_vector_json(&reports, synth_jaeckel_file);

    synth_jaeckel_file.close();
    delete(synth_parameters);
}

void synth_labels()
{
    Runners::SynthRunnerParameters* synth_parameters = get_synth_parameters();
    uint num_ones_arr[] = {4, 5, 6, 7, 8, 9, 10};

    std::vector<report_map> reports;
    for (auto num_ones: num_ones_arr)
    {
        Runners::SynthRunner synth_runner{};

        synth_parameters->num_ones = num_ones;

        synth_runner.set_parameters(synth_parameters);

        report_map jaeckel_report = synth_runner.labels(data_root, output_root);
        reports.push_back(jaeckel_report);
        print_report(&jaeckel_report);
    }

    std::ofstream synth_jaeckel_file;
    synth_jaeckel_file.open(reports_root + "/synth_labels.txt");
    save_report_vector_json(&reports, synth_jaeckel_file);

    synth_jaeckel_file.close();
    delete(synth_parameters);
}

void synth_cs_conf1()
{
    Runners::SynthRunnerParameters* synth_parameters = get_synth_parameters();
    uint mask_lengths[] = {12, 14};
    uint num_ones_arr[] = {4, 5, 6, 7, 8, 9, 10};
    uint cells_counts[] = {24};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto num_ones: num_ones_arr)
        {
            for (auto cells_count: cells_counts)
            {
                Runners::SynthRunner synth_runner{};

                synth_parameters->mask_length = mask_length;
                synth_parameters->num_ones = num_ones;
                synth_parameters->value_length = 150;
                synth_parameters->address_length = 150;
                synth_parameters->cells_count = cells_count*1000*1000;

                synth_runner.set_parameters(synth_parameters);

                report_map jaeckel_report = synth_runner.cs_conf1(data_root, output_root);
                reports.push_back(jaeckel_report);
                print_report(&jaeckel_report);
            }
        }
    }

    std::ofstream synth_jaeckel_file;
    synth_jaeckel_file.open(reports_root + "/synth_cs_conf1.txt");
    save_report_vector_json(&reports, synth_jaeckel_file);

    synth_jaeckel_file.close();
    delete(synth_parameters);
}

void synth_cs_conf2()
{
    Runners::SynthRunnerParameters* synth_parameters = get_synth_parameters();
    uint mask_lengths[] = {3};
    uint num_ones_arr[] = {4, 5, 6, 7, 8, 9, 10};
    uint cells_counts[] = {24};

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto num_ones: num_ones_arr)
        {
            for (auto cells_count: cells_counts)
            {
                Runners::SynthRunner synth_runner{};

                synth_parameters->mask_length = mask_length;
                synth_parameters->num_ones = num_ones;
                synth_parameters->value_length = 150;
                synth_parameters->address_length = 600;
                synth_parameters->cells_count = cells_count*1000*1000;

                synth_runner.set_parameters(synth_parameters);

                report_map jaeckel_report = synth_runner.cs_conf2(data_root, output_root);
                reports.push_back(jaeckel_report);
                print_report(&jaeckel_report);
            }
        }
    }

    std::ofstream synth_jaeckel_file;
    synth_jaeckel_file.open(reports_root + "/synth_cs_conf2.txt");
    save_report_vector_json(&reports, synth_jaeckel_file);

    synth_jaeckel_file.close();
    delete(synth_parameters);
}

void synth_cs_conf3()
{
    Runners::SynthRunnerParameters* synth_parameters = get_synth_parameters();
    uint mask_lengths[] = {8};
    uint num_ones_arr[] = {4, 5, 6, 7, 8, 9, 10};

    std::map<int, std::vector<std::pair<short, int>>> map = {
            {4,{
                {16, 55},
                //{15, 60},
                {14, 65},
                //{13, 70},
                {12, 75},
                //{11, 80},
                {10, 85},
                //{9, 90},
                {8, 95}
            }},
            {5,{
                {16, 45},
                //{15, 48},
                {14, 51},
                //{13, 54},
                {12, 57},
                //{11, 60},
                {10, 63},
                //{9, 66},
                {8, 69}
            }},
            {6,{
                {16, 37},
                //{15, 40},
                {14, 43},
                //{13, 46},
                {12, 49},
                //{11, 52},
                {10, 55},
                //{9, 58},
                {8, 61}
            }},
            {7,{{16, 32}, {14, 36}, {12, 40}, {10, 44}, {8, 48}}},
            {8,{{16, 28},{14, 32}, {12, 36}, {10, 40}, {8, 44}}},
            {9,{{16, 25}, {14, 27}, {12, 29}, {10, 31}, {8, 33}}},
            {10,{{16, 23}, {14, 25}, {12, 27}, {10, 29}, {8, 31}}},
    };

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto num_ones: num_ones_arr)
        {
            auto pairs = map[num_ones];
            for (auto pair: pairs)
            {
                auto coef = pair.first;
                auto cells_count = pair.second;
                auto m = coef*num_ones;

                Runners::SynthRunner synth_runner{};

                synth_parameters->mask_length = mask_length;
                synth_parameters->num_ones = num_ones;
                synth_parameters->value_length = m;
                synth_parameters->address_length = m;
                synth_parameters->cells_count = cells_count*1000*1000;
                synth_parameters->max_arrays = 10*1000;

                synth_runner.set_parameters(synth_parameters);

                report_map jaeckel_report = synth_runner.cs_conf3(data_root, output_root);
                reports.push_back(jaeckel_report);
                print_report(&jaeckel_report);
            }
        }
    }

    std::ofstream synth_jaeckel_file;
    synth_jaeckel_file.open(reports_root + "/synth_cs_conf3.txt");
    save_report_vector_json(&reports, synth_jaeckel_file);

    synth_jaeckel_file.close();
    delete(synth_parameters);
}

void synth_cs_conf4()
{
    Runners::SynthRunnerParameters* synth_parameters = get_synth_parameters();
    uint mask_lengths[] = {3};
    uint num_ones_arr[] = {4, 5, 6, 7, 8, 9, 10};

    std::map<int, std::vector<std::pair<short, int>>> map = {
//            {4,{{16, 55}, {14, 65}, {12, 75}, {10, 85}, {8, 95}, {6, 105}, {4, 115}}},
//            {5,{{16, 45}, {14, 51}, {12, 57}, {10, 63}, {8, 69}, {6, 75}, {4, 81}}},
            {6,{{16, 37}, {14, 43}, {12, 49}, {10, 55}, {8, 61}, {6, 67}, {4, 73}}},
            {7,{{16, 32}, {14, 36}, {12, 40}, {10, 44}, {8, 48}, {6, 52}, {4, 56}}},
            {8,{{16, 28}, {14, 32}, {12, 36}, {10, 40}, {8, 44}, {6, 48}, {4, 52}}},
            {9,{{16, 25}, {14, 27}, {12, 29}, {10, 31}, {8, 33}, {6, 35}, {4, 37}}},
            {10,{{16, 23}, {14, 25}, {12, 27}, {10, 29}, {8, 31}, {6, 33}, {4, 35}}},
    };

    std::vector<report_map> reports;
    for (auto mask_length: mask_lengths)
    {
        for (auto num_ones: num_ones_arr)
        {
            auto pairs = map[num_ones];
            for (auto pair: pairs)
            {
                auto coef = pair.first;
                if (coef <= 6)
                    continue;

                auto cells_count = pair.second;
                auto m = coef*num_ones;

                Runners::SynthRunner synth_runner{};

                synth_parameters->mask_length = mask_length;
                synth_parameters->num_ones = num_ones;
                synth_parameters->value_length = m;
                synth_parameters->address_length = 600;
                synth_parameters->cells_count = cells_count*1000*1000;
                synth_parameters->max_arrays = 25*1000;

                synth_runner.set_parameters(synth_parameters);

                report_map jaeckel_report = synth_runner.cs_conf4(data_root, output_root);
                reports.push_back(jaeckel_report);
                print_report(&jaeckel_report);
            }
        }
    }

    std::ofstream synth_jaeckel_file;
    synth_jaeckel_file.open(reports_root + "/synth_cs_conf4.txt");
    save_report_vector_json(&reports, synth_jaeckel_file);

    synth_jaeckel_file.close();
    delete(synth_parameters);
}


int main(int argc, char** argv)
{
    if (argc == 0)
        throw std::invalid_argument("Pass arguments!");

    // handle input arguments
    reports_root = argv[1];
    data_root = argv[2];
    output_root = argv[3];
    std::string experiment_num = argv[4];
    int experiment_num_int = std::stoi(experiment_num);
    if (experiment_num_int < 1 || experiment_num_int > 20)
        throw std::invalid_argument("Only {1,...,20} experiments are available now");

    typedef std::pair < std::string, std::function<void(void)>> test_type;
    std::vector<test_type> tests;
    tests.reserve(32);

    if (experiment_num_int == 1)
    {
        tests.emplace_back("Plain test with exact addresses", cifar10_naive);
        tests.emplace_back("Multiple write test", cifar10_multiple_write);
        tests.emplace_back("Iterative read test", cifar10_iterative_read);
        tests.emplace_back("Noisy address test", cifar10_noisy_address);
        tests.emplace_back("Noisy address and noisy value test", cifar10_noisy_address_noisy_value);
    }
    if (experiment_num_int == 2)
    {
//        tests.emplace_back( "Plain test with labels (stat)", labels_stat_naive );
//        tests.emplace_back( "Plain test with labels (bio)", labels_bio_naive );
        tests.emplace_back( "Plain test with labels (knots)", labels_knots );
    }
    if (experiment_num_int == 3)
    {
        //tests.emplace_back( "Plain test with matrix transformation (1)", cs1_naive_grid1 );
        //tests.emplace_back( "Plain test with matrix transformation (2)", cs1_naive_grid2 );
        tests.emplace_back( "Compressed sensing (image count grid)", cs1_image_count_grid );
    }
    if (experiment_num_int == 4)
    {
        //tests.emplace_back( "Plain test with matrix transformation (1)", cs1_naive_grid1 );
        //tests.emplace_back( "Plain test with matrix transformation (2)", cs1_naive_grid2 );
        tests.emplace_back( "Plain test with labels (knots)", labels_knots );
        tests.emplace_back( "Compressed sensing (image count grid)", cs1_image_count_grid );
    }
    if (experiment_num_int == 5)
    {
        //tests.emplace_back( "Plain test with matrix transformation (1)", cs1_naive_grid1 );
        //tests.emplace_back( "Plain test with matrix transformation (2)", cs1_naive_grid2 );
        tests.emplace_back( "Kanerva test", kanerva_run );
    }
    if (experiment_num_int == 6)
    {
        tests.emplace_back( "Plain test with labels (knots)", labels_knots );
    }
    if (experiment_num_int == 7)
    {
        tests.emplace_back( "Noisy test for CS SDM", cs1_noisy );
    }
    if (experiment_num_int == 8)
    {
        tests.emplace_back( "Noisy test for CS SDM (2 bits)", cs1_noisy_2 );
    }
    if (experiment_num_int == 9)
    {
        tests.emplace_back( "CS2 naive test", cs2_naive );
    }
    if (experiment_num_int == 10)
    {
        tests.emplace_back( "CS2 noisy test (1 feature dropped)", cs2_noisy_1 );
    }
    if (experiment_num_int == 11)
    {
        tests.emplace_back( "CS2 noisy test (2 features dropped)", cs2_noisy_2 );
    }
    if (experiment_num_int == 12)
    {
        tests.emplace_back( "CS2 naive test >= 3", cs2_naive_geq_3 );
    }
    if (experiment_num_int == 13)
    {
        tests.emplace_back( "CS2 naive test >= 3 S1", cs2_naive_geq_3_s1 );
    }
    if (experiment_num_int == 14)
    {
        tests.emplace_back( "CS2 naive test >= 3 S2", cs2_s2_naive_geq_3 );
    }
    if (experiment_num_int == 15)
    {
        tests.emplace_back( "Jaeckel synth", synth_jaeckel );
    }
    if (experiment_num_int == 16)
    {
        tests.emplace_back( "Labels synth", synth_labels );
    }
    if (experiment_num_int == 17)
    {
        tests.emplace_back( "CS config 1 synth", synth_cs_conf1 );
    }
    if (experiment_num_int == 18)
    {
        tests.emplace_back( "CS config 2 synth", synth_cs_conf2 );
    }
    if (experiment_num_int == 19)
    {
        tests.emplace_back( "CS config 3 synth", synth_cs_conf3 );
    }
    if (experiment_num_int == 20)
    {
        tests.emplace_back( "CS config 4 synth", synth_cs_conf4 );
    }

    std::cout.precision(6);

    for (const test_type& pair : tests)
    {
        std::cout << pair.first << std::endl << std::endl;
        pair.second();
//        try
//        {
//            pair.second();
//        }
//        catch (std::exception& e)
//        {
//            std::cout << "exception in test: " << e.what() << std::endl;
//            throw e;
//
//            system("pause");
//        }
        std::cout << std::endl;
    }

    system("pause");
}
#endif // !main_cu
