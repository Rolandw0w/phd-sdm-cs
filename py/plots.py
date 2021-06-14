import logging
import os

from matplotlib import pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s: %(message)s')
logger = logging.getLogger("METRICS")


def plot_kanerva(plots_path: str, kanerva_metrics_map: dict, kanerva_radius_list: list, image_nums: list,
                 p0: str = "0.990"):
    configs = [
        ("avg_l1", "Average L1", "Kanerva sparse signal reconstruction (avg L1)",
         f"kanerva_average_l1", np.arange(0, 15, 1)),
        ("fn_avg", "Average False Negative", "Kanerva sparse signal reconstruction  (avg FN)",
         f"kanerva_average_false_negative", np.arange(4, 5.01, 0.1)),
        ("fp_avg", "Average False Positive", "Kanerva sparse signal reconstruction  (avg FP)",
         f"kanerva_average_false_positive", np.arange(0, 10, 1)),
    ]
    for key, y_label, title, image_name, y_ticks in configs:

        x = image_nums
        x_ticks = x
        x_tick_labels = [str(k) for k in x_ticks]

        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D", "1"]
        for index, radius in enumerate(kanerva_radius_list):
            ys = np.array([kanerva_metrics_map[key][image_num][p0][radius] for image_num in image_nums])

            plt.plot(x, ys, marker=markers[index], color="k")

        plt.legend([f"Radius={radius}" for radius in kanerva_radius_list], fontsize=6)
        # plt.xlabel("Number of images restored")
        # plt.ylabel(y_label)
        # plt.title(title)

        plot_path = os.path.abspath(os.path.join(plots_path, f"{image_name}.png"))
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Image {plot_path} was saved")


def plot_labels(plots_path: str, labels_metrics_map: dict, labels_mask_range: list, image_nums: list):
    configs = [
        ("avg_l1", "Average L1", "Labels sparse signal reconstruction (avg L1)",
         f"labels_average_l1", np.arange(1, 12.01, 1)),
        ("fn_avg", "Average False Negative", "Labels sparse signal reconstruction  (avg FN)",
         f"labels_average_false_negative", np.arange(0, 5.51, 0.5)),
        ("fp_avg", "Average False Positive", "Labels sparse signal reconstruction  (avg FP)",
         f"labels_average_false_positive", np.arange(0, 8.01, 1)),
    ]
    for key, y_label, title, image_name, y_ticks in configs:

        x = image_nums
        x_ticks = x
        x_tick_labels = [str(k) for k in x_ticks]

        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        for index, mask_length in enumerate(labels_mask_range):
            ys = np.array([labels_metrics_map[key][image_num][mask_length] for image_num in image_nums])

            plt.plot(x, ys, marker=markers[index], color="k")

        plt.legend([f"mask_length={mask_length}" for mask_length in labels_mask_range], fontsize=8)
        # plt.xlabel("Number of images restored")
        # plt.ylabel(y_label)
        # plt.title(title)

        plot_path = os.path.abspath(os.path.join(plots_path, f"{image_name}.png"))
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Image {plot_path} was saved")


def plot_cs1(plots_path: str, cs1_metrics_map: dict, cs1_mask_range: list, image_nums: list):
    configs = [
        ("avg_l1", "Average L1", "CS1 sparse signal reconstruction (avg L1)",
         f"cs1_average_l1", np.arange(0, 3.01, 0.5)),
        ("fn_avg", "Average False Negative", "CS1 sparse signal reconstruction  (avg FN)",
         f"cs1_average_false_negative", np.arange(0, 2.01, 0.25)),
        ("fp_avg", "Average False Positive", "CS1 sparse signal reconstruction  (avg FP)",
         f"cs1_average_false_positive", np.arange(0, 2.01, 0.25)),
    ]
    for key, y_label, title, image_name, y_ticks in configs:

        x = image_nums
        x_ticks = x
        x_tick_labels = [str(k) for k in x_ticks]

        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(y_ticks, labels=y_tick_labels)
        markers = ["o", "x", "+", "s", "D"]
        for index, mask_length in enumerate(cs1_mask_range):
            ys = np.array([cs1_metrics_map[key][image_num][mask_length] for image_num in image_nums])

            plt.plot(x, ys, marker=markers[index], color="k")

        plt.legend([f"mask_length={mask_length}" for mask_length in cs1_mask_range], fontsize=6)
        # plt.xlabel("Number of images restored")
        # plt.ylabel(y_label)
        # plt.title(title)

        plot_path = os.path.abspath(os.path.join(plots_path, f"{image_name}.png"))
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Image {plot_path} was saved")


def plot_cs1_noisy_1(plots_path: str, cs1_metrics_map: dict, cs1_mask_range: list, image_nums: list):
    configs = [
        ("avg_l1", "Average L1", "CS1 noisy signal reconstruction 1 bit (avg L1)",
         f"cs1_average_l1_noisy_1", np.arange(0, 5.01, 0.5)),
        ("fn_avg", "Average False Negative", "CS1 noisy signal reconstruction 1 bit (avg FN)",
         f"cs1_average_false_negative_noisy_1", np.arange(0, 5.01, 0.5)),
        ("fp_avg", "Average False Positive", "CS1 noisy signal reconstruction 1 bit (avg FP)",
         f"cs1_average_false_positive_noisy_1", np.arange(0, 2.01, 0.25)),
    ]
    for key, y_label, title, image_name, y_ticks in configs:

        x = image_nums
        x_ticks = x
        x_tick_labels = [str(k) for k in x_ticks]

        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(y_ticks, labels=y_tick_labels)
        markers = [f"${mask_length}$" for mask_length in cs1_mask_range]
        for index, mask_length in enumerate(cs1_mask_range):
            ys = np.array([cs1_metrics_map[key][image_num][mask_length] for image_num in image_nums])

            plt.plot(x, ys, marker=markers[index], color="k")

        plt.legend([f"mask_length={mask_length}" for mask_length in cs1_mask_range], fontsize=6)
        # plt.xlabel("Number of images restored")
        # plt.ylabel(y_label)
        # plt.title(title)

        plot_path = os.path.abspath(os.path.join(plots_path, f"{image_name}.png"))
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Image {plot_path} was saved")


def plot_cs1_noisy_2(plots_path: str, cs1_metrics_map: dict, cs1_mask_range: list, image_nums: list):
    configs = [
        ("avg_l1", "Average L1", "CS1 noisy signal reconstruction 2 bit (avg L1)",
         f"cs1_average_l1_noisy_2", np.arange(0, 3.01, 0.5)),
        ("fn_avg", "Average False Negative", "CS1 noisy signal reconstruction 2 bit (avg FN)",
         f"cs1_average_false_negative_noisy_2", np.arange(0, 2.01, 0.25)),
        ("fp_avg", "Average False Positive", "CS1 noisy signal reconstruction 2 bit (avg FP)",
         f"cs1_average_false_positive_noisy_2", np.arange(0, 2.01, 0.25)),
    ]
    for key, y_label, title, image_name, y_ticks in configs:

        x = image_nums
        x_ticks = x
        x_tick_labels = [str(k) for k in x_ticks]

        y_tick_labels = [str(k) if isinstance(k, int) else ('%.2f' % k) for k in y_ticks]
        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(y_ticks, labels=y_tick_labels)
        markers = [f"${mask_length}$" for mask_length in cs1_mask_range]
        for index, mask_length in enumerate(cs1_mask_range):
            ys = np.array([cs1_metrics_map[key][image_num][mask_length] for image_num in image_nums])

            plt.plot(x, ys, marker=markers[index], color="k")

        plt.legend([f"mask_length={mask_length}" for mask_length in cs1_mask_range], fontsize=6)
        # plt.xlabel("Number of images restored")
        # plt.ylabel(y_label)
        # plt.title(title)

        plot_path = os.path.abspath(os.path.join(plots_path, f"{image_name}.png"))
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Image {plot_path} was saved")


def plot_comparison(plots_path: str, image_nums: list,
                    kanerva_metrics_map: dict, labels_metrics_map: dict, cs1_metrics_map: dict,
                    opt_kanerva_radius: int, opt_labels_mask_length: int, opt_cs1_mask_length: int,
                    kanerva_p0: str = "0.990",
                    ):
    configs = [
        ("avg_l1", "Average L1", "Kanerva SDM vs Jaeckel SDM vs CS SDM (avg L1)",
         f"comparison_average_l1", np.arange(0, 3.01, 0.5)),
        ("fn_avg", "Average False Negative", "Kanerva SDM vs Jaeckel SDM vs CS SDM (avg FN)",
         f"comparison_average_false_negative", np.arange(0, 2.01, 0.25)),
        ("fp_avg", "Average False Positive", "Kanerva SDM vs Jaeckel SDM vs CS SDM (avg FP)",
         f"comparison_average_false_positive", np.arange(0, 2.01, 0.25)),
    ]
    for key, y_label, title, image_name, y_ticks in configs:
        y_kanerva = [kanerva_metrics_map[key][image_num][kanerva_p0][opt_kanerva_radius] for image_num in image_nums]
        y_labels = [labels_metrics_map[key][image_num][opt_labels_mask_length] for image_num in image_nums]
        y_cs1 = [cs1_metrics_map[key][image_num][opt_cs1_mask_length] for image_num in image_nums]
        x = image_nums

        x_ticks = x
        x_tick_labels = [str(k) for k in x_ticks]

        y_min = np.floor(np.min([y_kanerva, y_labels, y_cs1]))
        y_max = np.ceil(np.max([y_kanerva, y_labels, y_cs1]))
        step = (y_max - y_min) / 20
        y_ticks = np.arange(y_min, y_max + 1e-10, step)
        y_tick_labels = ['%.2f' % k for k in y_ticks]

        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(y_ticks, labels=y_tick_labels)

        plt.plot(x, y_kanerva, marker="o", color="k")
        plt.plot(x, y_labels, marker="x", color="k")
        plt.plot(x, y_cs1, marker="+", color="k")

        plt.legend(["Kanerva SDM", "Jaeckel SDM", "CS SDM"])
        # plt.xlabel("Number of images restored")
        # plt.ylabel(y_label)
        # plt.title(title)
        plt.gray()

        plot_path = os.path.abspath(os.path.join(plots_path, f"{image_name}.png"))
        plt.savefig(plot_path, cmap="gray")
        plt.close()
        logger.info(f"Image {plot_path} was saved")


def plot_noisy_comparison(plots_path: str, image_nums: list,
                          kanerva_metrics_map: dict, labels_metrics_map: dict,
                          cs1_metrics_noisy_1_map: dict, cs1_metrics_noisy_2_map: dict,
                          opt_kanerva_radius: int, opt_labels_mask_length: int,
                          opt_cs1_noisy_1_mask_length: int, opt_cs1_noisy_2_mask_length: int,
                          image_nums_actual: list = None,
                          kanerva_p0: str = "0.990",
                          ):
    configs = [
        ("avg_l1", "Average L1", "Kanerva SDM vs Jaeckel SDM vs CS SDM (avg L1)",
         f"comparison_average_l1_noisy", np.arange(2, 7.51, 0.5)),
        ("fn_avg", "Average False Negative", "Kanerva SDM vs Jaeckel SDM vs CS SDM (avg FN)",
         f"comparison_average_false_negative_noisy", np.arange(0, 2.01, 0.25)),
        ("fp_avg", "Average False Positive", "Kanerva SDM vs Jaeckel SDM vs CS SDM (avg FP)",
         f"comparison_average_false_positive_noisy", np.arange(0, 2.01, 0.25)),
    ]
    for key, y_label, title, image_name, y_ticks in configs:
        y_kanerva = [kanerva_metrics_map[key][image_num][kanerva_p0][opt_kanerva_radius] for image_num in image_nums]
        y_labels = [labels_metrics_map[key][image_num][opt_labels_mask_length] for image_num in image_nums]
        y_cs1_noisy_1 = [cs1_metrics_noisy_1_map[key][image_num][opt_cs1_noisy_1_mask_length] for image_num in image_nums]
        y_cs1_noisy_2 = [cs1_metrics_noisy_2_map[key][image_num][opt_cs1_noisy_2_mask_length] for image_num in image_nums]
        x = image_nums_actual or image_nums

        x_ticks = image_nums
        x_tick_labels = [str(k) for k in x_ticks]

        y_min = np.floor(np.min([y_kanerva, y_labels, y_cs1_noisy_1]))
        y_max = np.ceil(np.max([y_kanerva, y_labels, y_cs1_noisy_2]))
        step = (y_max - y_min) / 20
        # y_ticks = np.arange(y_min, y_max + 1e-10, step)
        y_tick_labels = ['%.2f' % k for k in y_ticks]

        axes = plt.gca()
        axes.set_xlim([0, image_nums[-1] + image_nums[0]])
        axes.set_ylim([y_ticks[0], y_ticks[-1] + 1])

        plt.xticks(x_ticks, fontsize=6, labels=x_tick_labels)
        plt.yticks(y_ticks, labels=y_tick_labels)

        plot_data = [
            (y_kanerva, "$K$", "k", ":"),
            (y_labels, "$J$", "k", "-"),
            (y_cs1_noisy_1, "$1$", "k", "--"),
            (y_cs1_noisy_2, "$2$", "k", "-."),
        ]

        for y, marker, color, line_style in plot_data:
            plt.plot(x, y, color=color, linestyle=line_style)

        plt.legend(["Kanerva SDM", "Jaeckel SDM", "CS SDM 1 noisy bit", "CS SDM 2 noisy bits"])
        # plt.xlabel("Number of images restored")
        # plt.ylabel(y_label)
        # plt.title(title)
        plt.gray()

        plot_path = os.path.abspath(os.path.join(plots_path, f"{image_name}.png"))
        plt.savefig(plot_path, cmap="gray")
        plt.close()
        logger.info(f"Image {plot_path} was saved")


def plot_noisy_comparison_bars(plots_path: str, image_nums: list,
                          kanerva_metrics_map: dict, labels_metrics_map: dict,
                          cs1_metrics_noisy_1_map: dict, cs1_metrics_noisy_2_map: dict,
                          opt_kanerva_radius: int, opt_labels_mask_length: int,
                          opt_cs1_noisy_1_mask_length: int, opt_cs1_noisy_2_mask_length: int,
                          image_nums_actual: list = None,
                          kanerva_p0: str = "0.990",
                          ):
    configs = [
        ("avg_l1", "Average L1", "Kanerva SDM vs Jaeckel SDM vs CS SDM (avg L1)",
         f"comparison_average_l1_noisy_bars", np.arange(2, 7.01, 0.5)),
        ("fn_avg", "Average False Negative", "Kanerva SDM vs Jaeckel SDM vs CS SDM (avg FN)",
         f"comparison_average_false_negative_noisy_bars", np.arange(0, 2.01, 0.25)),
        ("fp_avg", "Average False Positive", "Kanerva SDM vs Jaeckel SDM vs CS SDM (avg FP)",
         f"comparison_average_false_positive_noisy_bars", np.arange(0, 2.01, 0.25)),
    ]
    for key, y_label, title, image_name, y_ticks in configs:
        y_kanerva = [kanerva_metrics_map[key][image_num][kanerva_p0][opt_kanerva_radius] for image_num in image_nums]
        y_labels = [labels_metrics_map[key][image_num][opt_labels_mask_length] for image_num in image_nums]
        y_cs1_noisy_1 = [cs1_metrics_noisy_1_map[key][image_num][opt_cs1_noisy_1_mask_length] for image_num in image_nums]
        y_cs1_noisy_2 = [cs1_metrics_noisy_2_map[key][image_num][opt_cs1_noisy_2_mask_length] for image_num in image_nums]

        x = np.array(image_nums_actual or image_nums)

        x_ticks = image_nums
        x_tick_labels = [str(k) for k in x_ticks]

        y_min = np.floor(np.min([y_kanerva, y_labels, y_cs1_noisy_1]))
        y_max = np.ceil(np.max([y_kanerva, y_labels, y_cs1_noisy_2]))
        step = (y_max - y_min) / 20
        # y_ticks = np.arange(y_min, y_max + 1e-10, step)
        y_tick_labels = ['%.2f' % k for k in y_ticks]

        width = 250  # the width of the bars

        fig, ax = plt.subplots()

        rect_kanerva = ax.bar(x - 3*width/2, y_kanerva, width, label="Kanerva SDM", hatch="*")
        rect_jaeckel = ax.bar(x - width/2, y_labels, width, label="Jaeckel SDM", hatch="+")
        rect_cs1_noisy_1 = ax.bar(x + width/2, y_cs1_noisy_1, width, label="CS SDM 1 noisy bit", hatch="/")
        rect_cs1_noisy_2 = ax.bar(x + 3*width/2, y_cs1_noisy_2, width, label="CS SDM 2 noisy bits", hatch="\\")

        ax.set_xlim([0, image_nums[-1]])
        ax.set_ylim([y_ticks[0], y_ticks[-1] + 1])

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

        ax.bar_label(rect_kanerva, padding=3)
        ax.bar_label(rect_jaeckel, padding=3)
        ax.bar_label(rect_cs1_noisy_1, padding=3)
        ax.bar_label(rect_cs1_noisy_2, padding=3)

        ax.legend()

        # ax.gray()
        # plt.xlabel("Number of images restored")
        # plt.ylabel(y_label)
        # plt.title(title)
        # plt.gray()

        plot_path = os.path.abspath(os.path.join(plots_path, f"{image_name}.png"))
        plt.savefig(plot_path, cmap="gray")
        plt.close()
        logger.info(f"Image {plot_path} was saved")

