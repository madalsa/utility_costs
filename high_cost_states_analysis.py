"""
Utility Cost Trends - High Electricity Cost States (IOUs)

Replication of CA electricity primer analysis for all IOUs in:
  Maine (ME), New York (NY), Massachusetts (MA), West Virginia (WV), Maryland (MD)

Data sources: EIA Form 861 (2005-2022), FERC Form 1
Plotting style follows RR.ipynb: indexed (base year = 2010), thick lines, clean grid.
"""

import matplotlib.pyplot as plt
import pandas as pd
import cpi
import numpy as np
from matplotlib.patches import Patch
import os
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

TARGET_STATES = ["ME", "NY", "MA", "WV", "MD"]
STATE_NAMES = {
    "ME": "Maine", "NY": "New York", "MA": "Massachusetts",
    "WV": "West Virginia", "MD": "Maryland",
}
BASE_YEAR = 2010  # Index base year for normalized plots

# Warm palette per utility (within‑state), cool palette for cross‑state
STATE_COLORS = {
    "ME": "#4477AA", "NY": "#EE6677", "MA": "#228833",
    "WV": "#CCBB44", "MD": "#AA3377",
}

NAME_ABBREV = {
    "Central Maine Power Co": "CMP", "Versant Power": "Versant",
    "Consolidated Edison Co-NY Inc": "ConEd",
    "Central Hudson Gas & Elec Corp": "CenHud",
    "New York State Elec & Gas Corp": "NYSEG",
    "Niagara Mohawk Power Corp.": "NiMo",
    "Orange & Rockland Utils Inc": "O&R",
    "Rochester Gas & Electric Corp": "RG&E",
    "Fishers Island Utility Co Inc": "FishIs",
    "Pennsylvania Electric Co": "PennElec",
    "NSTAR Electric Company": "NSTAR",
    "Massachusetts Electric Co": "MassElec",
    "Fitchburg Gas & Elec Light Co": "Fitchburg",
    "Nantucket Electric Co": "Nantucket",
    "Appalachian Power Co": "APCo", "Monongahela Power Co": "MonPwr",
    "The Potomac Edison Company": "PotEd", "Wheeling Power Co": "WhlPwr",
    "Black Diamond Power Co": "BlkDia",
    "Baltimore Gas & Electric Co": "BG&E",
    "Potomac Electric Power Co": "PEPCO", "Delmarva Power": "Delmarva",
}

OUTDIR = "HighCostStates_Outputs"
os.makedirs(OUTDIR, exist_ok=True)


# ── Plot style (matches RR.ipynb) ─────────────────────────────────────────────

def apply_plot_style():
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
    })

apply_plot_style()


def style_indexed_ax(ax, title="", ylabel="", xlim=(2006, 2023)):
    """Apply the RR.ipynb indexed‑plot style to an axes."""
    ax.axhline(y=1, linestyle="dashed", color="grey", lw=1)
    ax.axvline(x=BASE_YEAR, color="grey", lw=1)
    ax.grid(lw=0.3)
    ax.set_xlim(*xlim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 2))
    ax.tick_params(axis="x", rotation=90)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


# ── CPI pre‑computation ───────────────────────────────────────────────────────

def build_cpi_multipliers(years):
    """Return {year: multiplier_to_2022} dict."""
    return {int(yr): cpi.inflate(1.0, int(yr), to=2022) for yr in years}


# ── 1. Load EIA Form 861 ──────────────────────────────────────────────────────

def load_eia(cache_path="datafiles/Merged EIA data - high cost states IOUs.csv"):
    if os.path.isfile(cache_path):
        print(f"Loading cached EIA data from {cache_path}")
        return pd.read_csv(cache_path)

    yrs = list(range(2005, 2023))
    yr_dfs = []
    for yr in yrs:
        file = f"Sales_Ult_Cust_{yr}.xlsx"
        df = pd.read_excel(f"datafiles/EIA Data/{file}", sheet_name="States", header=2)
        df = df[(df["Ownership"] == "Investor Owned") & df["State"].isin(TARGET_STATES)]
        print(f"  {yr}: {len(df)} rows")

        try:
            df = df[["Data Year", "Utility Number", "Utility Name", "Service Type", "State",
                      "Thousand Dollars.1", "Megawatthours.1", "Count.1",
                      "Thousand Dollars.2", "Megawatthours.2", "Count.2",
                      "Thousand Dollars", "Megawatthours", "Count"]]
            num = ["Thousand Dollars.1", "Megawatthours.1", "Count.1",
                   "Thousand Dollars.2", "Megawatthours.2", "Count.2",
                   "Thousand Dollars", "Megawatthours", "Count"]
            df.loc[:, num] = df[num].apply(pd.to_numeric, errors="coerce").fillna(0)
            df = df.rename(columns={
                "Data Year": "Year", "Utility Number": "Utility ID",
                "Thousand Dollars": "Rev_res", "Megawatthours": "Sales_res", "Count": "Cust_res",
                "Thousand Dollars.1": "Rev_com", "Megawatthours.1": "Sales_com", "Count.1": "Cust_com",
                "Thousand Dollars.2": "Rev_ind", "Megawatthours.2": "Sales_ind", "Count.2": "Cust_ind",
            })
        except KeyError:
            df = df[["Data Year", "Utility Number", "Utility Name", "SERVICE_TYPE", "State",
                      "Thousands Dollars.1", "Megawatthours.1", "Count.1",
                      "Thousands Dollars.2", "Megawatthours.2", "Count.2",
                      "Thousands Dollars", "Megawatthours", "Count"]]
            num = ["Thousands Dollars.1", "Megawatthours.1", "Count.1",
                   "Thousands Dollars.2", "Megawatthours.2", "Count.2",
                   "Thousands Dollars", "Megawatthours", "Count"]
            df.loc[:, num] = df[num].apply(pd.to_numeric, errors="coerce").fillna(0)
            df = df.rename(columns={
                "Data Year": "Year", "SERVICE_TYPE": "Service Type", "Utility Number": "Utility ID",
                "Thousands Dollars": "Rev_res", "Megawatthours": "Sales_res", "Count": "Cust_res",
                "Thousands Dollars.1": "Rev_com", "Megawatthours.1": "Sales_com", "Count.1": "Cust_com",
                "Thousands Dollars.2": "Rev_ind", "Megawatthours.2": "Sales_ind", "Count.2": "Cust_ind",
            })
        yr_dfs.append(df.reset_index(drop=True))

    all_eia = pd.concat(yr_dfs, ignore_index=True)
    all_eia.to_csv(cache_path, index=False)
    print(f"Saved {len(all_eia)} rows → {cache_path}")
    return all_eia


# ── 2. Load FERC Form 1 ───────────────────────────────────────────────────────

TD_COLS = [
    "distribution_maintenance_expense_electric",
    "distribution_operation_expenses_electric",
    "transmission_maintenance_expense_electric",
    "transmission_operation_expense",
    "additions_transmission_plant",
    "additions_distribution_plant",
]


def load_ferc():
    ferc = pd.read_csv(
        "FERC1_datafiles/dispositions_and_opex_and_transmission_and_sales_and_rev_and_tdplant.csv"
    )
    eia_ferc = pd.read_csv(
        "FERC1_datafiles/eia ferc fuzzy matched manually corrected.csv"
    ).dropna(subset="eia")
    eia_codes = pd.read_csv("FERC1_datafiles/utilities_entity_eia.csv")
    ferc_codes = pd.read_csv("FERC1_datafiles/utilities_ferc1.csv")
    states = pd.read_csv("FERC1_datafiles/utilities_eia860.csv", usecols=[0, 2, 5])

    merged = pd.merge(eia_ferc, eia_codes, left_on="eia", right_on="utility_name_eia", how="left")
    merged = pd.merge(merged, ferc_codes, left_on="ferc", right_on="utility_name_ferc1", how="right")[
        ["utility_id_eia", "utility_id_ferc1", "utility_name_ferc1", "utility_name_eia"]
    ]
    state_utils = states[states["state"].isin(TARGET_STATES)][["utility_id_eia", "state"]].drop_duplicates()
    ferc_target = pd.merge(state_utils, merged, on="utility_id_eia", how="inner").dropna().drop_duplicates()

    ferc_data = pd.merge(ferc_target, ferc, on="utility_id_ferc1", how="inner")
    # Fix possible column suffix from merge
    name_col = [c for c in ferc_data.columns if c.startswith("utility_name_ferc1")][0]
    if name_col != "utility_name_ferc1":
        ferc_data = ferc_data.rename(columns={name_col: "utility_name_ferc1"})

    avail = [c for c in TD_COLS if c in ferc_data.columns]
    grouped = ferc_data.groupby(["state", "utility_name_ferc1", "report_year"])[avail].sum().reset_index()

    # Inflate to 2022$
    cpi_mult = build_cpi_multipliers(grouped["report_year"].unique())
    grouped["cpi_mult"] = grouped["report_year"].map(cpi_mult)
    for col in avail:
        grouped[f"real_{col}"] = grouped[col] * grouped["cpi_mult"]
    grouped.drop(columns="cpi_mult", inplace=True)

    # Derived columns: combined O&M
    grouped["real_dist_om"] = (
        grouped.get("real_distribution_maintenance_expense_electric", 0)
        + grouped.get("real_distribution_operation_expenses_electric", 0)
    )
    grouped["real_trans_om"] = (
        grouped.get("real_transmission_maintenance_expense_electric", 0)
        + grouped.get("real_transmission_operation_expense", 0)
    )
    grouped["real_total_om"] = grouped["real_dist_om"] + grouped["real_trans_om"]

    print(f"FERC data: {len(grouped)} rows, {grouped['report_year'].min()}-{grouped['report_year'].max()}")
    return grouped, avail


# ── 3. Figures ─────────────────────────────────────────────────────────────────

def fig1_snapshot(all_eia):
    """2022 snapshot bar chart: customers, sales, revenue by utility."""
    latest = all_eia[all_eia["Year"] == 2022].copy()
    latest["Short"] = latest["Utility Name"].map(NAME_ABBREV).fillna(latest["Utility Name"])

    snap = latest.groupby(["State", "Utility Name", "Short"]).agg({
        "Cust_res": "sum", "Sales_res": "sum", "Rev_res": "sum",
    }).reset_index()
    snap["order"] = snap["State"].map({s: i for i, s in enumerate(TARGET_STATES)})
    snap = snap.sort_values(["order", "Cust_res"], ascending=[True, False]).reset_index(drop=True)

    fig, axs = plt.subplots(1, 3, figsize=(28, 9))
    x = np.arange(len(snap))
    colors = [STATE_COLORS[s] for s in snap["State"]]

    metrics = [
        ("Cust_res", 1e6, "Residential Customers (millions)"),
        ("Sales_res", 1e6, "Residential Sales (TWh)"),
        ("Rev_res", 1e6, "Residential Revenue (billion $)"),
    ]
    for ax, (col, div, title) in zip(axs, metrics):
        ax.bar(x, snap[col] / div, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(snap["Short"], rotation=90)
        ax.set_title(title)
        ax.grid(lw=0.3)

    # state divider lines
    prev = None
    for i, s in enumerate(snap["State"]):
        if prev and s != prev:
            for ax in axs:
                ax.axvline(x=i - 0.5, color="k", ls="dotted", lw=2)
        prev = s

    legend = [Patch(facecolor=STATE_COLORS[s], label=STATE_NAMES[s]) for s in TARGET_STATES]
    axs[2].legend(handles=legend, loc="upper right")
    plt.suptitle("2022 Snapshot: IOUs in High Electricity Cost States", fontsize=20, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/Fig1_2022_Snapshot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Fig1 snapshot")


def fig2_residential_timeseries(all_eia):
    """State‑level residential time series (6 panels)."""
    cpi_mult = build_cpi_multipliers(all_eia["Year"].unique())

    total = all_eia.groupby(["State", "Year"]).agg({
        "Sales_res": "sum", "Rev_res": "sum", "Cust_res": "sum",
    }).reset_index()
    total["real_Rev_res"] = total.apply(lambda r: r["Rev_res"] * cpi_mult[r["Year"]], axis=1)

    fig, axs = plt.subplots(2, 3, figsize=(28, 18))
    for ax in axs.flatten():
        ax.grid(lw=0.3)

    for st in TARGET_STATES:
        d = total[total["State"] == st]
        c = STATE_COLORS[st]
        lbl = STATE_NAMES[st]
        axs[0, 0].plot(d["Year"], d["Cust_res"] / 1e6, color=c, lw=3, label=lbl)
        axs[0, 1].plot(d["Year"], d["Sales_res"] / 1e6, color=c, lw=3, label=lbl)
        axs[0, 2].plot(d["Year"], d["real_Rev_res"] / 1e6, color=c, lw=3, label=lbl)
        axs[1, 0].plot(d["Year"], d["real_Rev_res"] / d["Sales_res"], color=c, lw=3, label=lbl)
        axs[1, 1].plot(d["Year"], d["Sales_res"] / d["Cust_res"], color=c, lw=3, label=lbl)
        axs[1, 2].plot(d["Year"], d["real_Rev_res"] * 1e3 / d["Cust_res"], color=c, lw=3, label=lbl)

    titles = ["Customers (millions)", "Sales (TWh)", "Revenue (billion 2022$)",
              "Avg Price (2022$/kWh)", "Energy/Customer (MWh/yr)", "Revenue/Customer (2022$/yr)"]
    for ax, t in zip(axs.flatten(), titles):
        ax.set_title(t)
    axs[0, 0].legend()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Year")
    plt.suptitle("Residential IOU Trends — High Cost States (2005–2022)", fontsize=22, y=0.98)
    fig.savefig(f"{OUTDIR}/Fig2_Residential_TimeSeries.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Fig2 residential time series")


def fig2b_per_utility(all_eia):
    """Per‑utility breakdown within each state (one figure per state)."""
    cpi_mult = build_cpi_multipliers(all_eia["Year"].unique())

    for st in TARGET_STATES:
        d = all_eia[all_eia["State"] == st].copy()
        ut = d.groupby(["Utility Name", "Year"]).agg({
            "Cust_res": "sum", "Sales_res": "sum", "Rev_res": "sum",
        }).reset_index()
        ut["real_Rev_res"] = ut.apply(lambda r: r["Rev_res"] * cpi_mult[r["Year"]], axis=1)
        ut = ut[(ut["Cust_res"] > 0) & (ut["Sales_res"] > 0)]

        fig, axs = plt.subplots(2, 3, figsize=(28, 16))
        for ax in axs.flatten():
            ax.grid(lw=0.3)

        for name in sorted(ut["Utility Name"].unique()):
            u = ut[ut["Utility Name"] == name]
            lbl = NAME_ABBREV.get(name, name)
            axs[0, 0].plot(u["Year"], u["Cust_res"] / 1e6, lw=3, label=lbl)
            axs[0, 1].plot(u["Year"], u["Sales_res"] / 1e6, lw=3, label=lbl)
            axs[0, 2].plot(u["Year"], u["real_Rev_res"] / 1e6, lw=3, label=lbl)
            axs[1, 0].plot(u["Year"], u["real_Rev_res"] / u["Sales_res"], lw=3, label=lbl)
            axs[1, 1].plot(u["Year"], u["Sales_res"].values / u["Cust_res"].values, lw=3, label=lbl)
            axs[1, 2].plot(u["Year"], u["real_Rev_res"].values * 1e3 / u["Cust_res"].values, lw=3, label=lbl)

        titles = ["Customers (millions)", "Sales (TWh)", "Revenue (billion 2022$)",
                  "Avg Price (2022$/kWh)", "Energy/Customer (MWh/yr)", "Revenue/Customer (2022$/yr)"]
        for ax, t in zip(axs.flatten(), titles):
            ax.set_title(t)
        axs[0, 0].legend(fontsize=9)
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Year")
        plt.suptitle(f"{STATE_NAMES[st]} — Residential IOU Trends (2005–2022)", fontsize=22, y=0.98)
        fig.savefig(f"{OUTDIR}/Fig2b_{st}_PerUtility_Residential.png", dpi=300, bbox_inches="tight")
        plt.close()
    print("  ✓ Fig2b per‑utility plots (5 states)")


def fig5_td_indexed(ferc_grouped):
    """
    RR.ipynb‑style indexed T&D plots:
    - Per‑state figure with per‑utility subplots (dist O&M, trans O&M, total O&M)
    - Cross‑state comparison (indexed to 2010 = 1)
    """
    # ── Per‑utility, per‑state figures ──
    for st in TARGET_STATES:
        st_ferc = ferc_grouped[ferc_grouped["state"] == st]
        utils = sorted(st_ferc["utility_name_ferc1"].unique())
        if not utils:
            print(f"  ⚠ No FERC data for {STATE_NAMES[st]}")
            continue

        ncols = min(len(utils), 3)
        nrows = max(1, (len(utils) + ncols - 1) // ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows), squeeze=False)

        for idx, util in enumerate(utils):
            ax = axs[idx // ncols, idx % ncols]
            u = st_ferc[st_ferc["utility_name_ferc1"] == util].sort_values("report_year")
            short = util[:35] + "…" if len(util) > 35 else util

            for col, label, color in [
                ("real_dist_om", "Distribution O&M", "royalblue"),
                ("real_trans_om", "Transmission O&M", "darkorange"),
                ("real_total_om", "Total O&M", "saddlebrown"),
            ]:
                if col in u.columns:
                    base_row = u[u["report_year"] == BASE_YEAR]
                    if len(base_row) > 0 and base_row[col].item() != 0:
                        ax.plot(u["report_year"], u[col] / base_row[col].item(),
                                color=color, lw=4, label=label)

            for col, label, color in [
                ("real_additions_distribution_plant", "CapAdd‑Dist", "steelblue"),
                ("real_additions_transmission_plant", "CapAdd‑Trans", "coral"),
            ]:
                if col in u.columns:
                    base_row = u[u["report_year"] == BASE_YEAR]
                    if len(base_row) > 0 and base_row[col].item() != 0:
                        ax.plot(u["report_year"], u[col] / base_row[col].item(),
                                color=color, lw=2, ls="--", label=label)

            style_indexed_ax(ax, title=short, ylabel=f"Index ({BASE_YEAR} = 1)")

        for idx in range(len(utils), nrows * ncols):
            axs[idx // ncols, idx % ncols].set_visible(False)
        axs[0, min(len(utils), ncols) - 1].legend(fontsize=9, loc="upper left")
        plt.suptitle(f"T&D Expenses — {STATE_NAMES[st]} IOUs (indexed {BASE_YEAR} = 1)",
                     fontsize=20, y=1.02)
        plt.tight_layout()
        fig.savefig(f"{OUTDIR}/Fig5_{st}_TD_Indexed.png", dpi=300, bbox_inches="tight")
        plt.close()
    print("  ✓ Fig5 per‑state T&D indexed plots")


def fig5_cross_state_indexed(ferc_grouped):
    """Cross‑state comparison: total O&M indexed to 2010 = 1 (one line per state)."""
    agg = ferc_grouped.groupby(["state", "report_year"])[
        ["real_dist_om", "real_trans_om", "real_total_om"]
    ].sum().reset_index()

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    for col, ax, title in [
        ("real_dist_om", axs[0], "Distribution O&M"),
        ("real_trans_om", axs[1], "Transmission O&M"),
        ("real_total_om", axs[2], "Total O&M"),
    ]:
        for st in TARGET_STATES:
            d = agg[agg["state"] == st]
            base_row = d[d["report_year"] == BASE_YEAR]
            if len(base_row) == 0 or base_row[col].item() == 0:
                continue
            ax.plot(d["report_year"], d[col] / base_row[col].item(),
                    color=STATE_COLORS[st], lw=4, label=STATE_NAMES[st])
        style_indexed_ax(ax, title=title, ylabel=f"Index ({BASE_YEAR} = 1)")
    axs[0].legend(borderpad=0.1)
    plt.suptitle(f"O&M Expenses by State — IOU Comparison (indexed {BASE_YEAR} = 1)",
                 fontsize=20, y=1.04)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/Fig5_CrossState_OM_Indexed.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Fig5 cross‑state O&M indexed")


def fig5_cross_state_absolute(ferc_grouped):
    """Cross‑state absolute T&D expense levels (billion 2022$)."""
    agg = ferc_grouped.groupby(["state", "report_year"])[
        ["real_dist_om", "real_trans_om", "real_total_om",
         "real_additions_distribution_plant", "real_additions_transmission_plant"]
    ].sum().reset_index()

    labels = [
        ("real_dist_om", "Distribution O&M"),
        ("real_trans_om", "Transmission O&M"),
        ("real_total_om", "Total O&M"),
        ("real_additions_distribution_plant", "Capital Add — Distribution"),
        ("real_additions_transmission_plant", "Capital Add — Transmission"),
    ]
    # Only plot columns that exist
    labels = [(c, l) for c, l in labels if c in agg.columns]

    ncols = min(3, len(labels))
    nrows = (len(labels) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows), squeeze=False)

    for plot_idx, (col, title) in enumerate(labels):
        ax = axs[plot_idx // ncols, plot_idx % ncols]
        for st in TARGET_STATES:
            d = agg[agg["state"] == st]
            ax.plot(d["report_year"], d[col] / 1e9, color=STATE_COLORS[st], lw=4, label=STATE_NAMES[st])
        ax.grid(lw=0.3)
        ax.set_title(title)
        ax.set_ylabel("Billion 2022$")
    for idx in range(len(labels), nrows * ncols):
        axs[idx // ncols, idx % ncols].set_visible(False)

    axs[0, 0].legend(borderpad=0.1)
    plt.suptitle("T&D Expenses by State — Absolute Levels (billion 2022$)", fontsize=20, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/Fig5_CrossState_TD_Absolute.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Fig5 cross‑state T&D absolute")


# ── 4. Data export ─────────────────────────────────────────────────────────────

def export_data(all_eia, ferc_grouped, avail_cols):
    cpi_mult = build_cpi_multipliers(all_eia["Year"].unique())

    total = all_eia.groupby(["State", "Year"]).agg({
        "Sales_res": "sum", "Rev_res": "sum", "Cust_res": "sum",
        "Sales_com": "sum", "Rev_com": "sum", "Cust_com": "sum",
        "Sales_ind": "sum", "Rev_ind": "sum", "Cust_ind": "sum",
    }).reset_index()
    for suf in ["_res", "_com", "_ind"]:
        total[f"real_Rev{suf}"] = total.apply(lambda r: r[f"Rev{suf}"] * cpi_mult[r["Year"]], axis=1)
    total.to_csv(f"{OUTDIR}/EIA_state_level_timeseries.csv", index=False)
    all_eia.to_csv(f"{OUTDIR}/EIA_all_IOUs_by_utility.csv", index=False)
    ferc_grouped.to_csv(f"{OUTDIR}/FERC1_TD_expenses_high_cost_states.csv", index=False)
    print("  ✓ CSV exports")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("High‑Cost States IOU Analysis")
    print("=" * 60)

    print("\n[1/2] Loading EIA data …")
    all_eia = load_eia()
    print(f"  {len(all_eia)} rows, {all_eia['Year'].min()}–{all_eia['Year'].max()}, "
          f"{all_eia['Utility Name'].nunique()} IOUs across {sorted(all_eia['State'].unique())}")

    print("\n[2/2] Loading FERC data …")
    ferc_grouped, avail = load_ferc()

    print("\nGenerating figures …")
    fig1_snapshot(all_eia)
    fig2_residential_timeseries(all_eia)
    fig2b_per_utility(all_eia)
    fig5_td_indexed(ferc_grouped)
    fig5_cross_state_indexed(ferc_grouped)
    fig5_cross_state_absolute(ferc_grouped)

    print("\nExporting data …")
    export_data(all_eia, ferc_grouped, avail)

    print(f"\nDone. All outputs in {OUTDIR}/")


if __name__ == "__main__":
    main()
