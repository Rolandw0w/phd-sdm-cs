#ifndef main_cu
#define main_cu

#include <functional>
#include <iostream>
#include <map>

#include "main.cuh"


std::string reports_root_dir;
std::string data_root;


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
    const uint block_count = 32;
    const uint threads_per_block = 512;
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

Runners::LabelsRunnerParameters* get_labels_parameters(ReadingType reading_type, double bio_threshold)
{
    const uint image_count = 9000;
    const uint block_count = 32;
    const uint threads_per_block = 512;
    const uint mask_length = 2;
    const uint address_length = 600;
    const uint value_length = 600;
    const uint labels_count = 600;
    const uint cells_count = 3*1000*1000;
    auto* labels_parameters = new Runners::LabelsRunnerParameters(image_count, block_count, threads_per_block,
                                                                  mask_length, cells_count, address_length, value_length,
                                                                  labels_count, reading_type, bio_threshold);

    return labels_parameters;
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
    naive.open(reports_root_dir + "\\naive.txt");
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
    multiple_write.open(reports_root_dir + "\\cifar10_multiple_write.txt");
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
    iterative_read.open(reports_root_dir + "\\iterative_read.txt");
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
    noisy_address.open(reports_root_dir + "\\noisy_address.txt");
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
    noisy_address_noisy_value.open(reports_root_dir + "\\noisy_address_noisy_value.txt");
    save_report_vector_json(&reports, noisy_address_noisy_value);

    noisy_address_noisy_value.close();
    free(data);
}

void labels_stat_naive()
{
    const int image_num = 9000;
    const int labels_count = 651;

    bool** data = get_labels(labels_count, image_num, data_root);

    const double confidence = 0.9;
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

        report_map naive_report = labels_runner.naive(confidence);
        reports.push_back(naive_report);
        print_report(&naive_report);
    }
    std::ofstream labels_stat_naive;
    labels_stat_naive.open(reports_root_dir + "\\labels_stat_naive.txt");
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
    bool* cs1_data = get_cs1(labels_count, image_num, data_root);

    for (int i = 0; i < 4; i++)
    {
        std::cout << "lbl: ";
        for (int j = 0; j < labels_count; j++)
        {
            std::cout << data[i][j];
        }
        std::cout << std::endl;
        std::cout << "cs1: ";
        for (int j = 0; j < labels_count; j++)
        {
            std::cout << cs1_data[i*labels_count+j];
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }


    const double confidence = 0.9;

    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};
    Runners::LabelsRunnerParameters* labels_parameters = get_labels_parameters(ReadingType::STATISTICAL);

    std::vector<report_map> reports;
    for (uint image_count: image_counts)
    {
        Runners::LabelsRunner labels_runner{};
        labels_parameters->image_count = image_count;

        labels_runner.set_parameters(labels_parameters);
        labels_runner.set_data(&data);

        report_map naive_report = labels_runner.naive(confidence);
        reports.push_back(naive_report);
        print_report(&naive_report);
    }
    std::ofstream labels_stat_naive;
    labels_stat_naive.open(reports_root_dir + "\\labels_knots.txt");
    save_report_vector_json(&reports, labels_stat_naive);

    labels_stat_naive.close();
    free(data);
    delete(labels_parameters);
}

void labels_bio_naive()
{
    const int image_num = 9000;
    const int labels_count = 651;

    bool** data = get_labels(labels_count, image_num, data_root);

    const double confidence = 0.9;

    const double bio_thresholds[7] = {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99};

    Runners::LabelsRunnerParameters* labels_parameters = get_labels_parameters(ReadingType::BIOLOGICAL);

    std::vector<report_map> reports;
    for (double bio_threshold : bio_thresholds)
    {
        labels_parameters->bio_threshold = bio_threshold;

        Runners::LabelsRunner labels_runner{};
        labels_runner.set_parameters(labels_parameters);
        labels_runner.set_data(&data);

        report_map naive_report = labels_runner.naive(confidence);
        reports.push_back(naive_report);
        print_report(&naive_report);
    }

    std::ofstream labels_bio_naive;
    labels_bio_naive.open(reports_root_dir + "\\labels_bio_naive.txt");
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
//    std::cout << std::endl;
//    for (int i = 0; i < 1; i++)
//    {
//        for (int j = 0; j < labels_count; j++)
//        {
//            std::cout << data[j];
//        }
//    }
//    std::cout << std::endl;

    const double confidence = 0.9;

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

            report_map naive_report = cs1_runner.naive(confidence);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs1_naive;
    cs1_naive.open(reports_root_dir + "\\cs1_naive_grid1.txt");
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

    const double confidence = 0.9;

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

            report_map naive_report = cs1_runner.naive(confidence);
            reports.push_back(naive_report);
            print_report(&naive_report);
        }
    }

    std::ofstream cs1_naive;
    cs1_naive.open(reports_root_dir + "\\cs1_naive_grid2.txt");
    save_report_vector_json(&reports, cs1_naive);

    cs1_naive.close();
    free(data);
    delete(cs1_parameters);
}

void cs1_image_count_grid()
{
    const int image_num = 9000;
    const int labels_count = 600;

    bool* data = get_cs1(labels_count, image_num, data_root);

    const double confidence = 0.9;

    Runners::CS1RunnerParameters* cs1_parameters = get_cs1_parameters();
    uint image_counts[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000};

    std::vector<report_map> reports;
    for (auto images_read: image_counts)
    {
        Runners::CS1Runner cs1_runner{};

        cs1_parameters->images_read = images_read;

        cs1_runner.set_parameters(cs1_parameters);
        cs1_runner.set_data(&data);

        report_map naive_report = cs1_runner.naive(confidence);
        reports.push_back(naive_report);
        print_report(&naive_report);
    }

    std::ofstream cs1_naive;
    cs1_naive.open(reports_root_dir + "\\cs1_naive_grid2.txt");
    save_report_vector_json(&reports, cs1_naive);

    cs1_naive.close();
    free(data);
    delete(cs1_parameters);
}

int main(int argc, char** argv)
{
    if (argc == 0)
        throw std::invalid_argument("Pass arguments!");

    // handle input arguments
    reports_root_dir = argv[1];
    data_root = argv[2];
    std::string experiment_num = argv[3];
    int experiment_num_int = std::stoi(experiment_num);
    if (experiment_num_int < 1 || experiment_num_int > 3)
        throw std::invalid_argument("Only {1,2,3} experiments are available now");

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
