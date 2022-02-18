from pathlib import Path

from matplotlib import pyplot as plt

import pandas as pd

s_c_a = [
    (12, 8),
    (12, 12),
    (16, 8),
    (16, 12),
    (20, 8),
    (20, 12),
]

s_c_b = [
    (12, 6),
    (12, 8),
    (12, 12),
    (16, 6),
    (16, 8),
    (16, 12),
    (20, 6),
    (20, 8),
    (20, 12),
]


def a():
    l = []
    for s, c in s_c_a:
        ddf = pd.read_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_conf4/s_{s}_coef_{c}.csv")
        l += ddf.to_dict("records")
    l = [x for x in l if x["arrays_count"] not in {100_000, 250_000, 750_000, 1250_000, 1750_000}]
    sorted_l = sorted(l, key=lambda x: (x["features_count"], x["coefficient"]))
    df = pd.DataFrame(sorted_l)
    records = df.to_dict("records")
    sdm_type = "synth_k4_test"
    plot_dir = f"/home/rolandw0w/Development/PhD/output/synth/plots/{sdm_type}"
    df.to_csv(f"{plot_dir}/records.csv", index=False)

    s_list = sorted(list(df["features_count"].unique()))

    kanerva_df = pd.read_csv("/home/rolandw0w/Development/PhD/output/synth/kanerva/kanerva.csv")
    kanerva_records = kanerva_df.to_dict("records")
    jaeckel_df = pd.read_csv("/home/rolandw0w/Development/PhD/output/synth/jaeckel/jaeckel.csv")
    jaeckel_records = jaeckel_df.to_dict("records")

    x = list(range(500_000, 2_000_000 + 1, 500_000))
    x_labels = x
    markers = ["s", "+", "$J$", "$K$"]
    linestyles = ["-", "--", "-.", ":"]
    for s in s_list:
        k_records = [x for x in kanerva_records if x["features_count"] == s]
        j_records = [x for x in jaeckel_records if x["features_count"] == s]
        s_records = [x for x in records if x["features_count"] == s]
        coef_N_pairs = sorted(list(set([(x["coefficient"], x["cells_count"]) for x in s_records if x["coefficient"] != 6])), key=lambda x: x[0])

        for field in [
            "avg_fn", "avg_fn_noisy",
            "avg_fp", "avg_fp_noisy",
            "avg_l1", "avg_l1_noisy",
            "exact_percent", "exact_noisy_percent",
        ]:

            for index, (coef, N) in enumerate(coef_N_pairs):

                s_N_records = [x for x in s_records if x["cells_count"] == N and x["coefficient"] == coef and x["arrays_count"] in {500_000, 1000_000, 1500_000, 2000_000}]
                y = [x[field] for x in s_N_records]
                plt.plot(x, y, marker=markers[index], linestyle=linestyles[index])
                plt.xticks(x_labels, labels=[f"{round(xx / 1_000_000, 2)}M" for xx in x_labels], fontsize=12)

            #plt.yticks([0, *[pow(10, p) for p in range(-3, 3)]])

            y_kanerva = [x[field] for x in k_records if x["arrays_count"] in {500_000, 1000_000, 1500_000, 2000_000}]
            y_jeackel = [x[field] for x in j_records if x["arrays_count"] in {500_000, 1000_000, 1500_000, 2000_000}]

            plt.plot(x, y_jeackel, marker=markers[2], linestyle=linestyles[2])
            plt.plot(x, y_kanerva, marker=markers[3], linestyle=linestyles[3])

            if field not in {"exact_percent", "exact_noisy_percent"}:
                plt.yscale('symlog', linthreshy=1e-2 if s == 12 else 1e-6)
            else:
                plt.yscale('symlog', linthreshy=1e-6)
                plt.xlabel("Number of vectors")
                plt.ylabel("The percentage of correctly reconstructed vectors")

            # plt.legend([f"CS-SDM ({coef}*s, N={N//1_000_000}mln)" for coef, N in coef_N_pairs] +
            #            [f"Kanerva (R={s-4}, N=15mln)", "Jaeckel (K=4, N=15mln)"],
            #            prop={'size': 16})
            img_path = f"{plot_dir}/s{s}/{field}.jpg"
            plt.savefig(img_path)
            plt.close()
            from PIL import Image
            img = Image.open(img_path)#.resize((270, 180), Image.ANTIALIAS)
            img.save(img_path, "JPEG", quality=80)
            print(f"Saved {img_path}")


def b():
    l = []
    sdm_type = "synth_k4_small"
    plot_dir = f"/home/rolandw0w/Development/PhD/output/synth/plots/{sdm_type}"

    s_list = [12, 16, 20]

    cosamp_df = pd.read_csv("/output/synth/cs_conf4/cosamp_small.csv")
    linprog_df = pd.read_csv("/output/synth/cs_conf4/linprog_small.csv")

    records = sorted(cosamp_df.to_dict("records") + linprog_df.to_dict("records"),
                     key=lambda x: (x["restoration_type"], x["features_count"], x["coefficient"], x["arrays_count"]))

    xxx = list(range(20_000, 100_000 + 1, 20_000))
    x_labels = xxx
    for s in s_list:
        s_records = [x for x in records if x["features_count"] == s]
        coef_N_pairs = sorted(list(set([(x["coefficient"], x["cells_count"]) for x in s_records if x["coefficient"] != 12])), key=lambda x: x[0])

        for field in [
            "avg_fn", "avg_fn_noisy",
            "avg_fp", "avg_fp_noisy",
            "avg_l1", "avg_l1_noisy",
            "exact_percent", "exact_noisy_percent",
        ]:
            legend = []

            for coef, N in coef_N_pairs:
                for method, df in [("CoSaMP", cosamp_df), ("LinProg", linprog_df)]:
                    s_N_records = [x for x in s_records if x["cells_count"] == N and x["coefficient"] == coef and x["arrays_count"] in xxx and x["restoration_type"] == method.lower()]
                    y = [x[field] for x in s_N_records]
                    plt.plot(xxx, y, marker="o")
                    plt.xticks(x_labels, labels=[f"{round(xx / 1_000_000, 2)}M" for xx in x_labels], fontsize=6)
                    legend.append(f"CS SDM ({coef}*s, N={N//1_000_000}mln, {method})")

            plt.yscale('symlog', linthreshy=1e-1)
            yt = [0.0, 0.1, 0.2, 1, 2, 3, 4, 5, 6]
            plt.yticks(yt)

            plt.legend(legend)
            s_path = Path(f"{plot_dir}/s{s}")
            s_path.mkdir(exist_ok=True)
            img_path = s_path / f"{field}.png"
            plt.savefig(img_path)
            plt.close()
            print(f"Saved {img_path}")


a()


def c():
    sdm_type = "cs_mixed"
    plot_dir = f"/home/rolandw0w/Development/PhD/output/synth/plots/{sdm_type}"

    df = pd.read_csv("/home/rolandw0w/Development/PhD/output/synth/cs_conf4_mixed/cs_mixed.csv")

    records = sorted(df.to_dict("records"),
                     key=lambda x: (x["m"], x["restoration_type"], x["arrays_count"]))
    m_rest = sorted(list(set([(x["m"], x["restoration_type"]) for x in records])))

    xxx = list(range(30_000, 150_000 + 1, 30_000))
    x_labels = xxx

    for field in [
        "avg_fn", "avg_fn_noisy",
        "avg_fp", "avg_fp_noisy",
        "avg_l1", "avg_l1_noisy",
        "exact_percent", "exact_noisy_percent",
    ]:
        legend = []

        for m, restoration_type in m_rest:
            m_rest_records = [x for x in records if x["m"] == m and x["restoration_type"] == restoration_type and x["arrays_count"] in xxx]
            N = m_rest_records[0]["cells_count"]
            y = [x[field] for x in m_rest_records]
            plt.plot(xxx, y, marker="o")
            plt.xticks(x_labels, labels=[f"{xx}" for xx in x_labels], fontsize=6)
            legend.append(f"CS SDM (M={m}, N={N//1_000_000}mln, method={restoration_type})")

        # plt.yscale('symlog', linthreshy=1e-6)
        # yt = [0.0, 0.1, 0.2, 1, 2, 3, 4, 5, 6]
        # plt.yticks(yt)

        plt.legend(legend)
        s_path = Path(f"{plot_dir}")
        s_path.mkdir(exist_ok=True)
        img_path = s_path / f"{field}.png"
        plt.savefig(img_path)
        plt.close()
        print(f"Saved {img_path}")


c()


def d():
    from PIL import Image
    for s in [12, 16, 20]:
        for field in [
            "avg_fn", "avg_fn_noisy",
            "avg_fp", "avg_fp_noisy",
            "avg_l1", "avg_l1_noisy",
            "exact_percent", "exact_noisy_percent",
        ]:
            img = Image.open(f"/home/rolandw0w/Development/PhD/output/synth/plots/synth_k4_test/s{s}/{field}.jpg")
            r = img.convert('L')
            r.save(f"/home/rolandw0w/Development/PhD/output/synth/plots/synth_k4_test_bw/s{s}/{field}.jpg")


d()
