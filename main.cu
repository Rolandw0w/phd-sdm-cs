#ifndef main_cu
#define main_cu

#include <functional>
#include <iostream>
#include <map>

#include "main.cuh"


std::string reports_root_dir = "D:\\PhD\\Code\\reports";
//std::cout << reports_root_dir;

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

Runners::LabelsRunnerParameters* get_labels_parameters(ReadingType reading_type, double bio_threshold = 0.0)
{
    const uint image_count = 9000;
    const uint block_count = 32;
    const uint threads_per_block = 512;
    const uint mask_length = 2;
    const uint address_length = 651;
    const uint value_length = 651;
    const uint labels_count = 651;
    const uint cells_count = value_length*(value_length-1)/2;
    Runners::LabelsRunnerParameters* labels_parameters = new Runners::LabelsRunnerParameters(image_count, block_count, threads_per_block,
                                                                                             mask_length, cells_count, address_length, value_length, labels_count, reading_type, bio_threshold);

    return labels_parameters;
}

void print_report(report_map* report)
{
    for (auto elem : *report)
    {
        std::cout << elem.first << "=" << elem.second << std::endl;
    }
    std::cout << std::endl;
}

void print_report_json(report_map* report)
{
    std::cout << "{" << std::endl;
    for (auto elem : *report)
    {
        std::cout << "    \"" << elem.first << "\"" << ": " << elem.second << "," << std::endl;
    }
    std::cout << "}," << std::endl;
}

void save_report_vector_json(std::vector<report_map>* reports, std::ofstream& file)
{
    file << "[" << std::endl;
    for (auto report : *reports)
    {
        file << "    {" << std::endl;
        for (auto elem : report)
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
    bool** data = get_cifar10_images(cifar10_parameters->image_count);

    std::vector<report_map> reports;
    for (uint mask_length = min_mask_length; mask_length <= max_mask_length; mask_length += mask_length_step)
    {
        Runners::CIFAR10Runner cifar10_runner;
        cifar10_parameters->mask_length = mask_length;
        cifar10_runner.set_parameters(cifar10_parameters);
        cifar10_runner.set_data(&data);

        report_map naive_report = cifar10_runner.naive(confidence);
        reports.push_back(naive_report);
        print_report(&naive_report);
    }
    std::ofstream myfile;
    myfile.open(reports_root_dir + "\\naive.txt");
    save_report_vector_json(&reports, myfile);

    myfile.close();
    free(data);
}

void cifar10_multiple_write()
{
    const uint error_bits = 500;
    const uint min_write_count = 5;
    const uint max_write_count = 50;
    const uint write_count_step = 5;

    Runners::CIFAR10RunnerParameters* cifar10_parameters = get_cifar10_parameters();
    Runners::CIFAR10Runner cifar10_runner;
    cifar10_runner.set_parameters(cifar10_parameters);

    bool** data = get_cifar10_images(cifar10_parameters->image_count);
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
    std::ofstream myfile;
    myfile.open(reports_root_dir + "\\multiple_write.txt");
    save_report_vector_json(&reports, myfile);

    myfile.close();
    free(data);
}

void cifar10_iterative_read()
{
    const uint error_bits = 500;
    const uint min_iterations_count = 1;
    const uint max_iterations_count = 25;
    const uint iterations_count_step = 1;

    Runners::CIFAR10RunnerParameters* cifar10_parameters = get_cifar10_parameters();

    Runners::CIFAR10Runner cifar10_runner;
    cifar10_runner.set_parameters(cifar10_parameters);

    bool** data = get_cifar10_images(cifar10_parameters->image_count);
    cifar10_runner.set_data(&data);

    std::vector<report_map> reports;
    for (uint iterations_count = min_iterations_count; iterations_count <= max_iterations_count; iterations_count += iterations_count_step)
    {
        report_map iterative_read_report = cifar10_runner.iterative_read(iterations_count, error_bits);
        reports.push_back(iterative_read_report);
        print_report(&iterative_read_report);
    }
    std::ofstream myfile;
    myfile.open(reports_root_dir + "\\iterative_read.txt");
    save_report_vector_json(&reports, myfile);

    myfile.close();
    free(data);
}

void cifar10_noisy_address()
{
    const uint min_error_bits = 50;
    const uint max_error_bits = 500;
    const uint error_bits_step = 25;

    Runners::CIFAR10RunnerParameters* cifar10_parameters = get_cifar10_parameters();

    Runners::CIFAR10Runner cifar10_runner;
    cifar10_runner.set_parameters(cifar10_parameters);

    bool** data = get_cifar10_images(cifar10_parameters->image_count);
    cifar10_runner.set_data(&data);

    std::vector<report_map> reports;
    for (uint error_bits = min_error_bits; error_bits <= max_error_bits; error_bits += error_bits_step)
    {
        report_map noisy_address_report = cifar10_runner.noisy_address(error_bits);
        reports.push_back(noisy_address_report);
        print_report(&noisy_address_report);
    }
    std::ofstream myfile;
    myfile.open(reports_root_dir + "\\noisy_address.txt");
    save_report_vector_json(&reports, myfile);

    myfile.close();
    free(data);
}

void cifar10_noisy_address_noisy_value()
{
    const uint min_error_bits = 50;
    const uint max_error_bits = 500;
    const uint error_bits_step = 25;

    Runners::CIFAR10RunnerParameters* cifar10_parameters = get_cifar10_parameters();

    Runners::CIFAR10Runner cifar10_runner;
    cifar10_runner.set_parameters(cifar10_parameters);

    bool** data = get_cifar10_images(cifar10_parameters->image_count);
    cifar10_runner.set_data(&data);

    std::vector<report_map> reports;
    for (uint error_bits = min_error_bits; error_bits <= max_error_bits; error_bits += error_bits_step)
    {
        report_map noisy_address_report = cifar10_runner.noisy_address_noisy_value(error_bits);
        reports.push_back(noisy_address_report);
        print_report(&noisy_address_report);
    }
    std::ofstream myfile;
    myfile.open(reports_root_dir + "\\noisy_address_noisy_value.txt");
    save_report_vector_json(&reports, myfile);

    myfile.close();
    free(data);
}

void labels_stat_naive()
{
    const int image_num = 9000;
    const int labels_count = 651;

    bool** data = get_labels(labels_count, image_num);

    //for (int i = 0; i < image_num; i++)
    //{
    //	for (int j = 0; j < labels_count; j++)
    //	{
    //		std::cout << data[i][j];
    //	}
    //	std::cout << std::endl;
    //}

    const double confidence = 0.9;
    const uint min_mask_length = 2;
    const uint max_mask_length = 2;
    const uint mask_length_step = 1;

    Runners::LabelsRunnerParameters* labels_parameters = get_labels_parameters(ReadingType::STATISTICAL);

    std::vector<report_map> reports;
    for (uint mask_length = min_mask_length; mask_length <= max_mask_length; mask_length += mask_length_step)
    {
        Runners::LabelsRunner labels_runner;
        labels_parameters->mask_length = mask_length;
        labels_runner.set_parameters(labels_parameters);
        labels_runner.set_data(&data);

        report_map naive_report = labels_runner.naive(confidence);
        reports.push_back(naive_report);
        print_report(&naive_report);
    }
    std::ofstream myfile;
    myfile.open(reports_root_dir + "\\labels_stat_naive.txt");
    save_report_vector_json(&reports, myfile);

    myfile.close();
    free(data);
    delete(labels_parameters);
}

void labels_bio_naive()
{
    const int image_num = 9000;
    const int labels_count = 651;

    bool** data = get_labels(labels_count, image_num);

    //for (int i = 0; i < image_num; i++)
    //{
    //	for (int j = 0; j < labels_count; j++)
    //	{
    //		std::cout << data[i][j];
    //	}
    //	std::cout << std::endl;
    //}

    const double confidence = 0.9;
    const uint min_mask_length = 2;
    const uint max_mask_length = 2;
    const uint mask_length_step = 1;

    const double bio_thresholds[7] = {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99};

    Runners::LabelsRunnerParameters* labels_parameters = get_labels_parameters(ReadingType::BIOLOGICAL);

    std::vector<report_map> reports;
    for (double bio_threshold : bio_thresholds)
    {
        labels_parameters->bio_threshold = bio_threshold;

        Runners::LabelsRunner labels_runner;
        labels_runner.set_parameters(labels_parameters);
        labels_runner.set_data(&data);

        report_map naive_report = labels_runner.naive(confidence);
        reports.push_back(naive_report);
        print_report(&naive_report);
    }

    std::ofstream myfile;
    myfile.open(reports_root_dir + "\\labels_bio_naive.txt");
    save_report_vector_json(&reports, myfile);

    myfile.close();
    free(data);
    delete(labels_parameters);
}

int main()
{

    typedef std::pair < std::string, std::function<void(void)>> test_type;
    std::vector<test_type> tests;
    tests.reserve(50);

    //tests.push_back({ "Plain test with exact addresses", cifar10_naive });

    //tests.push_back({ "Multiple write test", cifar10_multiple_write });

    //tests.push_back({ "Iterative read test", cifar10_iterative_read });

    //tests.push_back({ "Noisy address test", cifar10_noisy_address});

    //tests.push_back({ "Noisy address and noisy value test", cifar10_noisy_address_noisy_value });

    tests.push_back({ "Plain test with labels (stat)", labels_stat_naive });

    tests.push_back({ "Plain test with labels (bio)", labels_bio_naive });

    std::cout.precision(6);

    for (test_type pair : tests)
    {
        std::cout << pair.first << std::endl << std::endl;
        try
        {
            pair.second();
        }
        catch (std::exception& e)
        {
            std::cout << "exception in test: " << e.what() << std::endl;

            system("pause");
        }
        std::cout << std::endl;
    }

    system("pause");
}
#endif // !main_cu