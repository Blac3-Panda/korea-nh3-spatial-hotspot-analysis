from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)

ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
PERCENTILES = [10, 15, 20]
K_VALUES = [3, 4, 5]
RANDOM_STATE = 42
FIGURE_DPI = 300
MAP_SIZE = (8, 10)
CHART_SIZE = (8.2, 4.8)


def configure_plot_style() -> str:
    """Apply consistent plotting style and a Korean-capable font."""
    preferred_fonts = [
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "AppleGothic",
        "DejaVu Sans",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    selected_font = next((name for name in preferred_fonts if name in available_fonts), "DejaVu Sans")

    plt.rcParams.update(
        {
            "font.family": selected_font,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
        }
    )
    return selected_font


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read CSV with common Korean encodings."""
    last_error = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV read failed for {path}: {last_error}")


def normalize_code(series: pd.Series) -> pd.Series:
    """Convert region codes to 5-digit strings."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d+)", expand=False).fillna("")
    s = s.str.zfill(5)
    return s


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert text number with comma separators to float."""
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "-": np.nan, "None": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")


def get_column(cols: List[str], candidates: List[str], contains: bool = False) -> str:
    for cand in candidates:
        for col in cols:
            if contains:
                if cand in col:
                    return col
            else:
                if cand == col:
                    return col
    raise KeyError(f"Column not found. candidates={candidates}, cols={cols}")


def discover_inputs(root: Path) -> Dict[str, Path]:
    shp_candidates = [p for p in root.rglob("*.shp") if p.name.lower() == "sig.shp"]
    if not shp_candidates:
        raise FileNotFoundError("SIG.shp not found under project root.")
    shp_path = sorted(shp_candidates, key=lambda p: len(str(p)))[0]

    csv_candidates = list(root.rglob("*.csv"))
    emission_path = None
    livestock_path = None
    emission_score = -1

    for csv_path in csv_candidates:
        try:
            df = read_csv_safely(csv_path)
        except Exception:
            continue

        cols = [str(c).strip() for c in df.columns]
        col_set = set(cols)

        is_emission = {"CD", "Region"}.issubset(col_set) and any("배출량" in c for c in cols)
        is_livestock = (
            {"CD", "Region", "dominant"}.issubset(col_set)
            and any("돼지" in c for c in cols)
            and any(c == "소" or "소" in c for c in cols)
        )

        if is_emission:
            score = 0
            if "2023" in csv_path.name:
                score += 10
            if "배출" in csv_path.name:
                score += 5
            if score > emission_score:
                emission_score = score
                emission_path = csv_path

        if is_livestock and livestock_path is None:
            livestock_path = csv_path

    if emission_path is None:
        raise FileNotFoundError("Emissions CSV not found.")
    if livestock_path is None:
        raise FileNotFoundError("Livestock CSV not found.")

    return {
        "shapefile": shp_path,
        "emissions_csv": emission_path,
        "livestock_csv": livestock_path,
    }


def load_and_merge(paths: Dict[str, Path]) -> gpd.GeoDataFrame:
    shp = gpd.read_file(paths["shapefile"])
    shp["SIG_CD"] = normalize_code(shp["SIG_CD"])

    emissions = read_csv_safely(paths["emissions_csv"])
    e_cols = [str(c).strip() for c in emissions.columns]
    e_cd = get_column(e_cols, ["CD"])
    e_region = get_column(e_cols, ["Region", "region"]) 
    e_value = get_column(e_cols, ["배출량"], contains=True)

    emissions = emissions.rename(columns={e_cd: "SIG_CD", e_region: "region", e_value: "배출량"})
    emissions["SIG_CD"] = normalize_code(emissions["SIG_CD"])
    emissions["배출량"] = to_numeric(emissions["배출량"])
    emissions = emissions[["SIG_CD", "region", "배출량"]]

    livestock = read_csv_safely(paths["livestock_csv"])
    l_cols = [str(c).strip() for c in livestock.columns]
    l_cd = get_column(l_cols, ["CD"])
    l_region = get_column(l_cols, ["Region", "region"]) 
    l_pig = get_column(l_cols, ["돼지"], contains=True)
    l_cow = get_column(l_cols, ["소"], contains=True)
    l_dom = get_column(l_cols, ["dominant"])

    livestock = livestock.rename(
        columns={
            l_cd: "SIG_CD",
            l_region: "region_livestock",
            l_pig: "돼지",
            l_cow: "소",
            l_dom: "dominant",
        }
    )
    livestock["SIG_CD"] = normalize_code(livestock["SIG_CD"])
    livestock["돼지"] = to_numeric(livestock["돼지"])
    livestock["소"] = to_numeric(livestock["소"])
    livestock = livestock[["SIG_CD", "region_livestock", "돼지", "소", "dominant"]]

    gdf = shp.merge(emissions, on="SIG_CD", how="left")
    gdf = gdf.merge(livestock, on="SIG_CD", how="left")

    return gdf


def feature_engineering(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    raw_area_km2 = gdf.geometry.area / 1_000_000
    gdf["area_km2"] = raw_area_km2.where(raw_area_km2 > 0, np.nan)

    gdf["density"] = gdf["배출량"] / gdf["area_km2"]
    gdf["density"] = gdf["density"].replace([np.inf, -np.inf], np.nan)
    gdf["log_density"] = np.log1p(gdf["density"].clip(lower=0))

    cent = gdf.geometry.centroid
    gdf["centroid_x"] = cent.x
    gdf["centroid_y"] = cent.y

    return gdf


def build_quality_report(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    total = len(gdf)
    raw_area = gdf.geometry.area / 1_000_000

    report_rows = [
        ("total_regions", total),
        ("missing_emissions_count", int(gdf["배출량"].isna().sum())),
        ("missing_emissions_ratio_pct", float(gdf["배출량"].isna().mean() * 100)),
        ("missing_livestock_count", int(gdf["dominant"].isna().sum())),
        ("missing_livestock_ratio_pct", float(gdf["dominant"].isna().mean() * 100)),
        ("area_zero_or_negative_count", int((raw_area <= 0).sum())),
        ("area_min_km2", float(raw_area.min())),
        ("area_p99_km2", float(raw_area.quantile(0.99))),
        ("area_max_km2", float(raw_area.max())),
    ]

    return pd.DataFrame(report_rows, columns=["metric", "value"])


def percentile_hotspots(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    gdf = gdf.copy()
    density_valid = gdf["density"].dropna()
    emission_total = gdf["배출량"].sum(skipna=True)

    rows = []
    for pct in PERCENTILES:
        quantile = 1 - (pct / 100)
        threshold = float(density_valid.quantile(quantile))
        col = f"hotspot_p{pct}"

        gdf[col] = (gdf["density"] >= threshold).fillna(False)
        hs_emission = gdf.loc[gdf[col], "배출량"].sum(skipna=True)
        hs_count = int(gdf[col].sum())

        rows.append(
            {
                "percentile_top": pct,
                "density_threshold": threshold,
                "hotspot_count": hs_count,
                "hotspot_ratio_pct": hs_count / len(gdf) * 100,
                "hotspot_emission_sum": float(hs_emission),
                "total_emission_sum": float(emission_total),
                "emission_contribution_pct": float(hs_emission / emission_total * 100) if emission_total > 0 else np.nan,
            }
        )

    return gdf, pd.DataFrame(rows)


def run_kmeans(gdf: gpd.GeoDataFrame, k_values: List[int]) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Dict[str, str]]:
    gdf = gdf.copy()
    summary_rows = []

    feature_sets = {
        "density_only": ["log_density"],
        "density_spatial": ["log_density", "centroid_x", "centroid_y"],
    }

    created_hotspot_flags: List[str] = []
    created_cluster_cols: List[str] = []

    for set_name, features in feature_sets.items():
        for k in k_values:
            cluster_col = f"cluster_{set_name}_k{k}"
            hotspot_col = f"hotspot_kmeans_{set_name}_k{k}"

            gdf[cluster_col] = pd.NA
            gdf[hotspot_col] = False

            mask = gdf[features + ["density", "배출량"]].notna().all(axis=1)
            if int(mask.sum()) < k:
                continue

            x = gdf.loc[mask, features].to_numpy()
            x_scaled = StandardScaler().fit_transform(x)

            km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
            labels = km.fit_predict(x_scaled)

            gdf.loc[mask, cluster_col] = labels
            cluster_stats = (
                gdf.loc[mask]
                .groupby(cluster_col, dropna=True)
                .agg(
                    region_count=(cluster_col, "size"),
                    mean_density=("density", "mean"),
                    median_density=("density", "median"),
                    emission_sum=("배출량", "sum"),
                )
                .reset_index()
            )

            hotspot_cluster = int(cluster_stats.sort_values("mean_density", ascending=False).iloc[0][cluster_col])
            gdf.loc[mask, hotspot_col] = gdf.loc[mask, cluster_col].astype(int) == hotspot_cluster

            total_emission_valid = gdf.loc[mask, "배출량"].sum()

            for _, row in cluster_stats.iterrows():
                cid = int(row[cluster_col])
                emission_sum = float(row["emission_sum"])
                summary_rows.append(
                    {
                        "feature_set": set_name,
                        "k": k,
                        "cluster_id": cid,
                        "is_hotspot_cluster": cid == hotspot_cluster,
                        "region_count": int(row["region_count"]),
                        "mean_density": float(row["mean_density"]),
                        "median_density": float(row["median_density"]),
                        "emission_sum": emission_sum,
                        "emission_contribution_pct": float(emission_sum / total_emission_valid * 100)
                        if total_emission_valid > 0
                        else np.nan,
                    }
                )

            created_cluster_cols.append(cluster_col)
            created_hotspot_flags.append(hotspot_col)

    default_cluster = "cluster_density_spatial_k4"
    default_hotspot = "hotspot_kmeans_density_spatial_k4"

    if default_cluster not in gdf.columns or gdf[default_cluster].notna().sum() == 0:
        if created_cluster_cols:
            default_cluster = created_cluster_cols[0]
            default_hotspot = created_hotspot_flags[0]
        else:
            raise RuntimeError("K-means failed for all configurations.")

    meta = {
        "default_cluster_col": default_cluster,
        "default_hotspot_col": default_hotspot,
    }

    return gdf, pd.DataFrame(summary_rows), meta


def dominant_group(value: object) -> str:
    if pd.isna(value):
        return "결측"
    text = str(value)
    if "돼지" in text or "pig" in text.lower():
        return "돼지"
    if "소" in text or "cow" in text.lower():
        return "소"
    return "기타"


def cross_analysis(gdf: gpd.GeoDataFrame, method_to_col: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for method, hotspot_col in method_to_col.items():
        tmp = gdf[[hotspot_col, "dominant_group", "density", "배출량"]].copy()
        tmp["hotspot_status"] = np.where(tmp[hotspot_col], "hotspot", "non_hotspot")

        status_total = tmp.groupby("hotspot_status").size().to_dict()
        avg_density = tmp.groupby("hotspot_status")["density"].mean().to_dict()
        total_emission = tmp["배출량"].sum(skipna=True)
        emission_share = (tmp.groupby("hotspot_status")["배출량"].sum() / total_emission * 100).to_dict()

        counts = tmp.groupby(["hotspot_status", "dominant_group"]).size().reset_index(name="count")
        for _, r in counts.iterrows():
            hs = r["hotspot_status"]
            rows.append(
                {
                    "method": method,
                    "hotspot_status": hs,
                    "dominant_group": r["dominant_group"],
                    "count": int(r["count"]),
                    "ratio_within_status_pct": float(r["count"] / status_total.get(hs, 1) * 100),
                    "avg_density_by_status": float(avg_density.get(hs, np.nan)),
                    "emission_contribution_by_status_pct": float(emission_share.get(hs, np.nan)),
                }
            )

    return pd.DataFrame(rows)


def save_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    out_png: Path,
    legend_label: str,
    cmap: str = "YlOrRd",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=MAP_SIZE)
    gdf.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        legend=True,
        linewidth=0.25,
        edgecolor="#F8F8F8",
        legend_kwds={"label": legend_label, "orientation": "horizontal", "shrink": 0.55, "pad": 0.02},
        missing_kwds={"color": "#E8E8E8", "label": "결측"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.08)
    fig.savefig(out_png, dpi=FIGURE_DPI)
    plt.close(fig)


def save_hotspot_map(gdf: gpd.GeoDataFrame, flag_col: str, title: str, out_png: Path) -> None:
    data = gdf.copy()
    data["_label"] = np.where(data[flag_col], "Hotspot", "기타")

    colors = {"Hotspot": "#D1492F", "기타": "#BCC3CA"}

    fig, ax = plt.subplots(1, 1, figsize=MAP_SIZE)
    for label, color in colors.items():
        sub = data[data["_label"] == label]
        if len(sub) > 0:
            sub.plot(ax=ax, color=color, linewidth=0.25, edgecolor="#F8F8F8")

    handles = [Patch(facecolor=colors[k], edgecolor="black", label=k) for k in colors]
    ax.legend(handles=handles, loc="lower left")
    ax.set_facecolor("#EEF1F4")
    ax.set_title(title)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.08)
    fig.savefig(out_png, dpi=FIGURE_DPI)
    plt.close(fig)


def save_dominant_map(gdf: gpd.GeoDataFrame, out_png: Path) -> None:
    data = gdf.copy()
    data["dominant_map"] = data["dominant_group"].replace({"결측": "기타"})

    colors = {"돼지": "#D95C8A", "소": "#6E4A35", "기타": "#BCC3CA"}
    order = ["돼지", "소", "기타"]

    fig, ax = plt.subplots(1, 1, figsize=MAP_SIZE)
    for label in order:
        sub = data[data["dominant_map"] == label]
        if len(sub) > 0:
            sub.plot(ax=ax, color=colors[label], linewidth=0.25, edgecolor="#F8F8F8")

    handles = [Patch(facecolor=colors[k], edgecolor="black", label=k) for k in order if (data["dominant_map"] == k).any()]
    ax.legend(handles=handles, loc="lower left")
    ax.set_facecolor("#EEF1F4")
    ax.set_title("돼지 vs 소 우세 지역")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.08)
    fig.savefig(out_png, dpi=FIGURE_DPI)
    plt.close(fig)


def save_hotspot_dominant_map(gdf: gpd.GeoDataFrame, hotspot_col: str, out_png: Path) -> None:
    data = gdf.copy()
    data["hotspot_dominant"] = "비핫스팟"
    data.loc[data[hotspot_col] & (data["dominant_group"] == "돼지"), "hotspot_dominant"] = "핫스팟-돼지"
    data.loc[data[hotspot_col] & (data["dominant_group"] == "소"), "hotspot_dominant"] = "핫스팟-소"
    data.loc[data[hotspot_col] & (~data["dominant_group"].isin(["돼지", "소"])), "hotspot_dominant"] = "핫스팟-기타"

    colors = {
        "핫스팟-돼지": "#D95C8A",
        "핫스팟-소": "#6E4A35",
        "핫스팟-기타": "#8E8E8E",
        "비핫스팟": "#BCC3CA",
    }
    order = ["핫스팟-돼지", "핫스팟-소", "핫스팟-기타", "비핫스팟"]

    fig, ax = plt.subplots(1, 1, figsize=MAP_SIZE)
    for label in order:
        sub = data[data["hotspot_dominant"] == label]
        if len(sub) > 0:
            sub.plot(ax=ax, color=colors[label], linewidth=0.25, edgecolor="#F8F8F8")

    handles = [Patch(facecolor=colors[k], edgecolor="black", label=k) for k in order if (data["hotspot_dominant"] == k).any()]
    ax.legend(handles=handles, loc="lower left")
    ax.set_facecolor("#EEF1F4")
    ax.set_title("Hotspot 내 돼지 vs 소 우세")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.08)
    fig.savefig(out_png, dpi=FIGURE_DPI)
    plt.close(fig)


def save_density_hist(gdf: gpd.GeoDataFrame, out_png: Path) -> None:
    vals = gdf["density"].dropna()
    vals = vals[vals > 0]

    if len(vals) == 0:
        return

    bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 36)
    fig, ax = plt.subplots(figsize=CHART_SIZE)
    ax.hist(vals, bins=bins, color="#2C7FB8", edgecolor="white", alpha=0.92)
    median = float(np.median(vals))
    p85 = float(np.quantile(vals, 0.85))
    ax.axvline(median, color="#1B4F72", linestyle="--", linewidth=1.5, label=f"중앙값 {median:,.0f}")
    ax.axvline(p85, color="#C0392B", linestyle="-.", linewidth=1.5, label=f"상위 15% 경계 {p85:,.0f}")
    ax.set_xscale("log")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_xlabel("배출밀도 (kg/km²/yr, 로그축)")
    ax.set_ylabel("시군구 수")
    ax.set_title("배출밀도 분포 (오른쪽일수록 고밀도)")
    ax.legend(loc="upper left")
    fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.15)
    fig.savefig(out_png, dpi=FIGURE_DPI)
    plt.close(fig)


def save_dominant_bar(gdf: gpd.GeoDataFrame, hotspot_col: str, out_png: Path) -> None:
    tmp = gdf[[hotspot_col, "dominant_group"]].copy()
    tmp["status"] = np.where(tmp[hotspot_col], "Hotspot", "비핫스팟")

    dist = pd.crosstab(tmp["status"], tmp["dominant_group"], normalize="index") * 100
    status_order = ["Hotspot", "비핫스팟"]
    col_order = [c for c in ["돼지", "소", "기타", "결측"] if c in dist.columns]
    dist = dist.reindex(status_order).fillna(0.0)
    if col_order:
        dist = dist[col_order]

    fig, ax = plt.subplots(figsize=CHART_SIZE)
    dist.plot(kind="bar", stacked=True, ax=ax, width=0.48, color=["#4C78A8", "#F58518", "#54A24B", "#9E9E9E"])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_ylabel("비율 (%)")
    ax.set_xlabel("")
    ax.set_title("우세 축종 비율")
    ax.set_xticklabels([str(x.get_text()) for x in ax.get_xticklabels()], rotation=0, ha="center")
    ax.legend(title="dominant", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(left=0.12, right=0.82, top=0.88, bottom=0.15)
    fig.savefig(out_png, dpi=FIGURE_DPI)
    plt.close(fig)


def build_kpi_line(percentile_table: pd.DataFrame, cross_table: pd.DataFrame) -> str:
    p15 = percentile_table[percentile_table["percentile_top"] == 15].iloc[0]
    p15_contrib = float(p15["emission_contribution_pct"])

    hs = cross_table[(cross_table["method"] == "percentile_p15") & (cross_table["hotspot_status"] == "hotspot")]
    hs = hs.sort_values("ratio_within_status_pct", ascending=False)

    if len(hs) > 0:
        dom = str(hs.iloc[0]["dominant_group"])
        dom_ratio = float(hs.iloc[0]["ratio_within_status_pct"])
        dom_text = f"{dom} {dom_ratio:.1f}%"
    else:
        dom_text = "N/A"

    return f"상위 15% 기여율: {p15_contrib:.1f}% | 핫스팟 우세 축종: {dom_text}"


def run_pipeline(root: Path, out_dir: Path) -> None:
    out_fig = out_dir / "figures"
    out_tbl = out_dir / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tbl.mkdir(parents=True, exist_ok=True)

    paths = discover_inputs(root)
    selected_font = configure_plot_style()

    print("[INPUTS]")
    for key, path in paths.items():
        print(f"- {key}: {path}")
    print(f"- plot_font: {selected_font}")

    gdf = load_and_merge(paths)
    quality = build_quality_report(gdf)

    gdf = feature_engineering(gdf)
    gdf, pct_table = percentile_hotspots(gdf)
    gdf, kmeans_table, kmeans_meta = run_kmeans(gdf, K_VALUES)

    gdf["dominant_group"] = gdf["dominant"].map(dominant_group)

    analysis_methods = {
        "percentile_p15": "hotspot_p15",
        "kmeans_default": kmeans_meta["default_hotspot_col"],
    }
    cross_table_df = cross_analysis(gdf, analysis_methods)

    save_choropleth(gdf, "배출량", "시군구 NH3 배출량", out_fig / "map_emissions_kgyr.png", "배출량 (kg/yr)")
    save_choropleth(gdf, "density", "시군구 NH3 배출밀도", out_fig / "map_density_kgkm2yr.png", "배출밀도 (kg/km²/yr)")
    save_hotspot_map(gdf, "hotspot_p15", "상위 15% Hotspot", out_fig / "map_hotspot_percentile_p15.png")
    save_hotspot_map(
        gdf,
        kmeans_meta["default_hotspot_col"],
        "K-means Hotspot",
        out_fig / "map_hotspot_kmeans_default.png",
    )
    save_dominant_map(gdf, out_fig / "map_dominant_pig_vs_cow.png")
    save_hotspot_dominant_map(gdf, "hotspot_p15", out_fig / "map_hotspot_dominant_pig_vs_cow.png")
    save_density_hist(gdf, out_fig / "hist_density_logscale.png")
    save_dominant_bar(gdf, "hotspot_p15", out_fig / "bar_dominant_hotspot_vs_nonhotspot.png")

    pct_table.to_csv(out_tbl / "percentile_hotspot_contribution.csv", index=False, encoding="utf-8-sig")
    kmeans_table.to_csv(out_tbl / "kmeans_cluster_summary.csv", index=False, encoding="utf-8-sig")
    cross_table_df.to_csv(out_tbl / "hotspot_dominant_cross.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(out_tbl / "data_quality_report.csv", index=False, encoding="utf-8-sig")

    export_cols = [
        "SIG_CD",
        "SIG_KOR_NM",
        "배출량",
        "area_km2",
        "density",
        "log_density",
        "hotspot_p10",
        "hotspot_p15",
        "hotspot_p20",
        kmeans_meta["default_cluster_col"],
        kmeans_meta["default_hotspot_col"],
        "돼지",
        "소",
        "dominant",
        "dominant_group",
    ]
    available_cols = [c for c in export_cols if c in gdf.columns]
    gdf.drop(columns=["geometry"]).loc[:, available_cols].to_csv(
        out_tbl / "region_level_results.csv", index=False, encoding="utf-8-sig"
    )

    kpi_line = build_kpi_line(pct_table, cross_table_df)
    (out_dir / "kpi_summary.txt").write_text(kpi_line + "\n", encoding="utf-8")

    print("\n[KPI]")
    print(kpi_line)

    print("\n[OUTPUTS]")
    print(f"- figures: {out_fig}")
    print(f"- tables: {out_tbl}")
    print(f"- kpi: {out_dir / 'kpi_summary.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Korea SIG NH3 hotspot analysis pipeline")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root path")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "outputs", help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.root, args.output)


if __name__ == "__main__":
    main()
