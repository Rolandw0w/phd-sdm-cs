#ifndef labels_runner_cu
#define labels_runner_cu


#include <iostream>
#include <fstream>

#include "labels_runner.cuh"


namespace Runners
{
	report_map LabelsRunner::naive(const std::string& data_path, const std::string& output_path)
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
		report.insert({"reading_type", (int) parameters->reading_type});

		if (parameters->reading_type == ReadingType::BIOLOGICAL)
			report.insert({"bio_threshold", parameters->bio_threshold});

		SDM_LABELS<short, short, int> sdm(parameters->mask_length, parameters->address_length,
			parameters->value_length, parameters->cells_count, parameters->block_count,
			parameters->threads_per_block, parameters->reading_type, parameters->bio_threshold);

		long write_time_start = clock();

		for (uint i = 0; i < parameters->images_read; i++)
		{
			sdm.write(data[i]);
		}

		long write_time = clock() - write_time_start;

		report.insert({"avg_write_time", (double)write_time / parameters->images_read});

		double sum = 0.0;

		long read_time_start = clock();

		unsigned long all_bits = parameters->labels_count * parameters->images_read;
		long incorrect_bits = 0;
		long correct100 = 0;

		int tp = 0;
		int fp = 0;
		int tn = 0;
		int fn = 0;
		int one_diff = 0;

        std::ofstream restored;

        auto mode = parameters->reading_type == ReadingType::BIOLOGICAL ? "bio" : "stat";
        restored.open(output_path + "/labels_" + mode + "_K_" + std::to_string(parameters->mask_length) +
                      "_I_" + std::to_string(parameters->images_read) + ".csv");

        for (uint i = 0; i < parameters->images_read; i++)
        {
            bool* remembered = sdm.read(data[i]);
            int dist = hamming_distance(data[i], remembered, parameters->labels_count);

            for (uint j = 0; j < parameters->labels_count; j++)
            {
                //std::cout << remembered[j];
                auto sep = (j == parameters->value_length - 1) ? "\n" : ",";
                restored << remembered[j] << sep;
                if (data[i][j] == 1 && remembered[j] == 1)
                    tp += 1;
                if (data[i][j] == 0 && remembered[j] == 1)
                    fp += 1;
                if (data[i][j] == 0 && remembered[j] == 0)
                    tn += 1;
                if (data[i][j] == 1 && remembered[j] == 0)
                    fn += 1;
            }
            //std::cout << std::endl;

            if (dist == 0)
                correct100 += 1;
            else if (dist == 1)
            {
                one_diff += 1;
            }

            incorrect_bits += dist;

            sum += dist;

            free(remembered);

        }
        restored.close();
        long read_time = clock() - read_time_start;

        report.insert({"avg_read_time", (double)read_time / parameters->images_read});
        report.insert({"avg_dist", sum / parameters->images_read});
        report.insert({"incorrect_bits", incorrect_bits});
        report.insert({"correct_bits", all_bits - incorrect_bits});
        report.insert({"all_bits", all_bits});
        report.insert({"correct_100", correct100});
        report.insert({"correct_bits_percent", (100 * (double)(all_bits - incorrect_bits) / all_bits)});
//        report.insert({"min_activations", sdm.get_min_activations()});
//        report.insert({"max_activations", sdm.get_max_activations()});
//        report.insert({"activated_cells_count", sdm.get_activations_num()});
        report.insert({"tp", tp});
        report.insert({"tp_avg", (double)tp / parameters->images_read});
        report.insert({"fp", fp});
        report.insert({"fp_avg", (double)fp / parameters->images_read});
        report.insert({"tn", tn});
        report.insert({"tn_avg", (double)tn / parameters->images_read});
        report.insert({"fn", fn});
        report.insert({"fn_avg", (double)fn / parameters->images_read});
        report.insert({"non_readable", sdm.get_non_readable()});
        report.insert({"non_writable", sdm.get_non_writable()});
        report.insert({"one_diff", one_diff});

        //sdm.~SDM_LABELS();

        return report;
    }
}

#endif // labels_runner_cu