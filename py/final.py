from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from PIL import Image


plot_dir = Path("/home/rolandw0w/Development/PhD/output/synth/plots/final")
markers = ["s", "+", "o", "D"]
linestyles = ["-", "--", "-.", ":"]
legend_size = 16
x_ticks_size = 12
width = 640
height = 480


def plot_kanerva():
    kanerva_df = pd.read_csv("/home/rolandw0w/Development/PhD/output/synth/kanerva/kanerva_comp.csv")
    kanerva_records = kanerva_df.to_dict("records")

    x = list(range(500_000, 2_000_000 + 1, 500_000))
    x_labels = x
    s = 16
    k_records = [x for x in kanerva_records if x["features_count"] == s]
    radius_N_pairs = sorted(list(set([(x["radius"], x["cells_count"]) for x in k_records])), key=lambda x: x[0])

    for field in [
        "avg_fn", "avg_fn_noisy",
        "avg_fp", "avg_fp_noisy",
        "avg_l1", "avg_l1_noisy",
        "exact_percent", "exact_noisy_percent",
    ]:

        for index, (radius, N) in enumerate(radius_N_pairs):
            s_N_records = [x for x in k_records if x["cells_count"] == N and x["radius"] == radius and x["arrays_count"] in {500_000, 1000_000, 1500_000, 2000_000}]
            y = [x[field] for x in s_N_records]
            plt.plot(x, y, c="k", marker=markers[index], linestyle=linestyles[index])
            plt.xticks(x_labels, labels=[f"{round(xx / 1_000_000, 2)}M" for xx in x_labels], fontsize=x_ticks_size)

        # plt.yscale('symlog', linthresh=1)

        plt.legend([f"Kanerva (radius={radius}, N={N//1_000_000}mln)" for radius, N in radius_N_pairs],
                   prop={'size': legend_size})
        img_path = f"{plot_dir}/kanerva/{field}.png"
        plt.savefig(img_path)
        plt.close()

        img = Image.open(img_path)
        img = img.resize((width, height), Image.ANTIALIAS)
        img.convert("L").save(img_path)
        print(f"Saved {img_path}")


def plot_jaeckel():
    jaeckel_df = pd.read_csv("/home/rolandw0w/Development/PhD/output/synth/jaeckel/jaeckel_comp.csv")
    jaeckel_records = jaeckel_df.to_dict("records")

    x = list(range(500_000, 2_000_000 + 1, 500_000))
    x_labels = x
    s = 16
    k_records = [x for x in jaeckel_records if x["features_count"] == s]
    radius_N_pairs = sorted(list(set([(x["mask_length"], x["cells_count"]) for x in k_records])), key=lambda x: x[0])

    for field in [
        "avg_fn", "avg_fn_noisy",
        "avg_fp", "avg_fp_noisy",
        "avg_l1", "avg_l1_noisy",
        "exact_percent", "exact_noisy_percent",
    ]:

        for index, (K, N) in enumerate(radius_N_pairs):
            s_N_records = [x for x in k_records if x["cells_count"] == N and x["mask_length"] == K and x["arrays_count"] in {500_000, 1000_000, 1500_000, 2000_000}]
            y = [x[field] for x in s_N_records]
            plt.plot(x, y, c="k", marker=markers[index], linestyle=linestyles[index])
            plt.xticks(x_labels, labels=[f"{round(xx / 1_000_000, 2)}M" for xx in x_labels], fontsize=x_ticks_size)

        # plt.yscale('symlog', linthresh=1)

        plt.legend([f"Jaeckel (mask_length=={K}, N={N//1_000_000}mln)" for K, N in radius_N_pairs],
                   prop={'size': legend_size})
        img_path = f"{plot_dir}/jaeckel/{field}.png"
        plt.savefig(img_path)
        plt.close()

        img = Image.open(img_path)
        img = img.resize((width, height), Image.ANTIALIAS)
        img.convert("L").save(img_path)
        print(f"Saved {img_path}")


def plot_cs():
    cs_records = []
    for s in [16]:
        for coef in [8, 12]:
            df = pd.read_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s_{s}_coef_{coef}.csv")
            cs_records += df.to_dict("records")

    x = list(range(500_000, 2_000_000 + 1, 500_000))
    x_labels = x
    coef_N_pairs = sorted(list(set([(x["coefficient"], x["cells_count"]) for x in cs_records])), key=lambda x: x[0])

    for field in [
        "avg_fn", "avg_fn_noisy",
        "avg_fp", "avg_fp_noisy",
        "avg_l1", "avg_l1_noisy",
        "exact_percent", "exact_noisy_percent",
    ]:

        for index, (coef, N) in enumerate(coef_N_pairs):
            s_N_records = [x for x in cs_records if x["cells_count"] == N and x["coefficient"] == coef and x["arrays_count"] in {500_000, 1000_000, 1500_000, 2000_000}]
            y = [x[field] for x in s_N_records]
            plt.plot(x, y, c="k", marker=markers[index], linestyle=linestyles[index])
            plt.xticks(x_labels, labels=[f"{round(xx / 1_000_000, 2)}M" for xx in x_labels], fontsize=x_ticks_size)

        # plt.yscale('symlog', linthresh=1)

        plt.legend([f"CS SDM (M={coef*s}, N={N//1_000_000}mln)" for coef, N in coef_N_pairs],
                   prop={'size': legend_size})
        img_path = f"{plot_dir}/cs/{field}.png"
        plt.savefig(img_path)
        plt.close()

        img = Image.open(img_path)
        img = img.resize((width, height), Image.ANTIALIAS)
        img.convert("L").save(img_path)
        print(f"Saved {img_path}")


def plot_cs_mini():
    cosamp_df = pd.read_csv("/home/rolandw0w/Development/PhD/output/synth/plots/final/cs_cosamp_small.csv")
    linprog_df = pd.read_csv("/home/rolandw0w/Development/PhD/output/synth/plots/final/cs_linprog_small.csv")

    records = sorted(cosamp_df.to_dict("records") + linprog_df.to_dict("records"),
                     key=lambda x: (x["restoration_type"], x["features_count"], x["coefficient"], x["arrays_count"]))

    xxx = list(range(20_000, 100_000 + 1, 20_000))
    x_labels = xxx
    s = 16
    s_records = [x for x in records if x["features_count"] == s]
    coef_N_pairs = sorted(list(set([(x["coefficient"], x["cells_count"]) for x in s_records if x["coefficient"] != 12])), key=lambda x: x[0])

    for field in [
        "avg_fn", "avg_fn_noisy",
        "avg_fp", "avg_fp_noisy",
        "avg_l1", "avg_l1_noisy",
        "exact_percent", "exact_noisy_percent",
    ]:
        legend = []

        i = 0
        for coef, N in coef_N_pairs:
            for method, df in [("CoSaMP", cosamp_df), ("LinProg", linprog_df)]:
                s_N_records = [x for x in s_records if x["cells_count"] == N and x["coefficient"] == coef and x["arrays_count"] in xxx and x["restoration_type"] == method.lower()]
                y = [x[field] for x in s_N_records]
                plt.plot(xxx, y, marker=markers[i], linestyle=linestyles[i])
                i += 1
                plt.xticks(x_labels, labels=[f"{round(xx / 1_000_000, 2)}M" for xx in x_labels], fontsize=x_ticks_size)
                legend.append(f"CS SDM (M={coef*s}, N={N//1_000_000}mln, {method})")

        plt.yscale('symlog', linthresh=5e-3)

        plt.legend(legend, prop={'size': legend_size})
        img_path = plot_dir / f"cs_small/{field}.png"
        plt.savefig(img_path)
        plt.close()

        img = Image.open(img_path)
        img = img.resize((width, height), Image.ANTIALIAS)
        img.convert("L").save(img_path)
        print(f"Saved {img_path}")


def plot_comparison():
    cs_records = []
    for s in [16]:
        for coef in [8, 12]:
            df = pd.read_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s_{s}_coef_{coef}.csv")
            cs_records += df.to_dict("records")

    jaeckel_records = pd.read_csv("/home/rolandw0w/Development/PhD/output/synth/jaeckel/jaeckel_comp.csv").to_dict("records")
    j_records = [x for x in jaeckel_records if x["mask_length"] == 4]
    kanerva_records = pd.read_csv("/home/rolandw0w/Development/PhD/output/synth/kanerva/kanerva_comp.csv").to_dict("records")
    k_records = [x for x in kanerva_records if x["radius"] == 12]

    x = list(range(500_000, 2_000_000 + 1, 500_000))
    x_labels = x
    coef_N_pairs = sorted(list(set([(x["coefficient"], x["cells_count"]) for x in cs_records])), key=lambda x: x[0])

    for field in [
        "avg_fn", "avg_fn_noisy",
        "avg_fp", "avg_fp_noisy",
        "avg_l1", "avg_l1_noisy",
        "exact_percent", "exact_noisy_percent",
    ]:

        for index, (coef, N) in enumerate(coef_N_pairs):
            s_N_records = [x for x in cs_records if x["cells_count"] == N and x["coefficient"] == coef and x["arrays_count"] in {500_000, 1000_000, 1500_000, 2000_000}]
            y = [x[field] for x in s_N_records]
            plt.plot(x, y, c="k", marker=markers[index], linestyle=linestyles[index])
            plt.xticks(x_labels, labels=[f"{round(xx / 1_000_000, 2)}M" for xx in x_labels], fontsize=x_ticks_size)

        plt.yscale('symlog', linthresh=1e-2)

        y_kanerva = [x[field] for x in k_records if x["arrays_count"] in {500_000, 1000_000, 1500_000, 2000_000}]
        y_jeackel = [x[field] for x in j_records if x["arrays_count"] in {500_000, 1000_000, 1500_000, 2000_000}]

        plt.plot(x, y_kanerva, c="k", marker="$K$", linestyle=linestyles[2])
        plt.plot(x, y_jeackel, c="k", marker="$J$", linestyle=linestyles[3])

        plt.legend([f"CS SDM (M={coef*s}, N={N//1_000_000}mln)" for coef, N in coef_N_pairs] +
                   [f"Kanerva (radius=12, N=15mln)"] +
                   [f"Jaeckel (mask_length==4, N=15mln)"],
                   prop={'size': legend_size})
        img_path = f"{plot_dir}/comparison/{field}.png"
        plt.savefig(img_path)
        plt.close()

        img = Image.open(img_path)
        img = img.resize((width, height), Image.ANTIALIAS)
        img.convert("L").save(img_path)
        print(f"Saved {img_path}")


plot_kanerva()
plot_jaeckel()
plot_cs()
plot_cs_mini()
plot_comparison()
