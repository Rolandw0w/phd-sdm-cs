#ifndef labels_runner_cu
#define labels_runner_cu


#include <iostream>
#include <fstream>

#include "labels_runner.cuh"


namespace Runners
{
	report_map LabelsRunner::naive(const double confidence, const bool save_images, const std::string& images_path)
	{
		report_map report;
		const int required_exact_bits = (int) (parameters->value_length * confidence);

		report.insert({"mask_length", parameters->mask_length});
		report.insert({"value_length", parameters->value_length});
		report.insert({"address_length", parameters->address_length});
		report.insert({"cells_count", parameters->cells_count});
		report.insert({"image_count", parameters->image_count});
		report.insert({"block_count", parameters->block_count});
		report.insert({"threads_per_block", parameters->threads_per_block});
		report.insert({"labels_count", parameters->labels_count});
		report.insert({"reading_type", (int) parameters->reading_type});
		report.insert({"confidence", confidence});

		if (parameters->reading_type == ReadingType::BIOLOGICAL)
			report.insert({"bio_threshold", parameters->bio_threshold});

		SDM_LABELS<short, short, short> sdm(parameters->mask_length, parameters->address_length,
			parameters->value_length, parameters->cells_count, parameters->block_count,
			parameters->threads_per_block, parameters->reading_type, parameters->bio_threshold);

		long write_time_start = clock();

		for (uint i = 0; i < parameters->image_count; i++)
		{
			sdm.write(data[i]);
		}

		long write_time = clock() - write_time_start;

		report.insert({"avg_write_time", (double)write_time / parameters->image_count});

		double sum = 0.0;
		int exact_num = 0;

		long read_time_start = clock();

		unsigned long all_bits = parameters->labels_count * parameters->image_count;
		long incorrect_bits = 0;
		long correct100 = 0;

		int tp = 0;
		int fp = 0;
		int tn = 0;
		int fn = 0;
		int one_diff = 0;

		std::ofstream prediction;
		std::string predict_home = R"(D:\PhD\Code\predicts\)";
		std::ofstream input;
		std::string input_home = R"(D:\PhD\Code\predicts\input.txt)";

        if (parameters->reading_type == ReadingType::BIOLOGICAL)
        {
            predict_home.append("bio_");
            predict_home.append(std::to_string(parameters->bio_threshold));
            predict_home.append(".txt");
        } else {
            predict_home.append("stat.txt");
        }
        prediction.open(predict_home);
        input.open(input_home);
        for (uint i = 0; i < parameters->image_count; i++)
        {
            bool* remembered = sdm.read(data[i]);
            for (int j = 0; j < parameters->labels_count; j++)
            {
                std::string to_write_pred = remembered[j] ? "1" : "0";
                std::string to_write_true = data[i][j] ? "1" : "0";
                prediction << to_write_pred;
                input << to_write_true;
            }
            prediction << std::endl;
            input << std::endl;
            int dist = hamming_distance(data[i], remembered, parameters->labels_count);

            for (uint j = 0; j < parameters->labels_count; j++)
            {
                if (data[i][j] == 1 && remembered[j] == 1)
                    tp += 1;
                if (data[i][j] == 0 && remembered[j] == 1)
                    fp += 1;
                if (data[i][j] == 0 && remembered[j] == 0)
                    tn += 1;
                if (data[i][j] == 1 && remembered[j] == 0)
                    fn += 1;
            }

            if (dist == 0)
                correct100 += 1;
            else if (dist == 1)
            {
                one_diff += 1;
            }

            incorrect_bits += dist;

            uint exact_bits = parameters->labels_count - dist;
            exact_num += (exact_bits >= required_exact_bits) ? 1 : 0;
            sum += dist;

            if (save_images)
            {
                std::string image_number = std::to_string(i + 1);

                std::string input_path = images_path;
                input_path.append(image_number);
                input_path.append("_input.bmp");
                save_image_bmp(_strdup(input_path.c_str()), 32, 32, data[i]);

                std::string output_path = images_path;
                output_path.append("_output.bmp");
                save_image_bmp(_strdup(output_path.c_str()), 32, 32, remembered);
            }

            free(remembered);

        }
        prediction.close();
        input.close();
        long read_time = clock() - read_time_start;

        report.insert({"avg_read_time", (double)read_time / parameters->image_count});
        report.insert({"avg_dist", sum / parameters->image_count});
        report.insert({"exact_num_paper", exact_num});
        report.insert({"paper_percent", (100.0 * exact_num / parameters->image_count)});
        report.insert({"incorrect_bits", incorrect_bits});
        report.insert({"correct_bits", all_bits - incorrect_bits});
        report.insert({"all_bits", all_bits});
        report.insert({"correct_100", correct100});
        report.insert({"correct_bits_percent", (100 * (double)(all_bits - incorrect_bits) / all_bits)});
//        report.insert({"min_activations", sdm.get_min_activations()});
//        report.insert({"max_activations", sdm.get_max_activations()});
//        report.insert({"activated_cells_count", sdm.get_activations_num()});
        report.insert({"tp", tp});
        report.insert({"tp_avg", (double)tp / parameters->image_count});
        report.insert({"fp", fp});
        report.insert({"fp_avg", (double)fp / parameters->image_count});
        report.insert({"tn", tn});
        report.insert({"tn_avg", (double)tn / parameters->image_count});
        report.insert({"fn", fn});
        report.insert({"fn_avg", (double)fn / parameters->image_count});
        report.insert({"non_readable", sdm.get_non_readable()});
        report.insert({"non_writable", sdm.get_non_writable()});
        report.insert({"one_diff", one_diff});

        sdm.~SDM_LABELS();

        return report;
    }
}

#endif // labels_runner_cu