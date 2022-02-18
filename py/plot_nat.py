from matplotlib import pyplot as plt

import pandas as pd


sdm_type = "nat"

cs_df = pd.read_csv(f"/home/rolandw0w/Development/PhD/output/synth/cs_nat.csv")
cs_records = cs_df.to_dict("records")

jaeckel_df = pd.read_csv(f"/home/rolandw0w/Development/PhD/output/synth/jaeckel_nat.csv")
jaeckel_records = jaeckel_df.to_dict("records")

kanerva_df = pd.read_csv(f"/home/rolandw0w/Development/PhD/output/synth/kanerva_nat.csv")
kanerva_records = kanerva_df.to_dict("records")

x = list(range(250, 5_001, 250))
x_labels = list(range(250, 5_001, 250))

ms = sorted(list(set([x["m"] for x in cs_records])))

kanerva = kanerva_records
jaeckel = jaeckel_records

for field in [
    "avg_fn", "avg_fn_noisy",
    "avg_fp", "avg_fp_noisy",
    "avg_l1", "avg_l1_noisy",
    "exact_percent", "exact_noisy_percent",
]:
    y_kanerva = [x[field] for x in kanerva]
    plt.plot(x, y_kanerva)

    y_jaeckel = [x[field] for x in jaeckel]
    plt.plot(x, y_jaeckel)

    for m in ms:
        m_records = [x for x in cs_records if x["m"] == m and x["restoration_type"] == "linprog"]
        y = [x[field] for x in m_records]
        plt.plot(x, y)
        plt.xticks(x_labels, labels=[str(xx) for xx in x_labels], fontsize=6)

    plt.legend(["Kanerva", "Jaeckel"] + [f"CS SDM (m={m}*s)" for m in ms])
    img_path = f"/home/rolandw0w/Development/PhD/output/synth/plots/{sdm_type}/{field}.png"
    plt.savefig(img_path)
    plt.close()
    print(f"Saved {img_path}")
