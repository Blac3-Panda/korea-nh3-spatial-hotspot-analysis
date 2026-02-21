
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
from scipy.stats import chi2_contingency
from esda import Moran, Moran_Local
from libpysal.weights import Queen, Rook, w_subset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)

ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
PERCENTILES = [10, 15, 20]
K_VALUES = [3, 4, 5]
DEFAULT_K = 4
RANDOM_STATE = 42
FIGURE_DPI = 320
MAP_SIZE = (8, 10)
CHART_SIZE = (8.2, 4.8)

SPATIAL_SIDO_MAP = {
    "11": "Seoul", "26": "Busan", "27": "Daegu", "28": "Incheon", "29": "Gwangju",
    "30": "Daejeon", "31": "Ulsan", "36": "Sejong", "41": "Gyeonggi", "42": "Gangwon",
    "43": "Chungbuk", "44": "Chungnam", "45": "Jeonbuk", "46": "Jeonnam", "47": "Gyeongbuk",
    "48": "Gyeongnam", "50": "Jeju",
}
np.random.seed(RANDOM_STATE)


def configure_plot_style() -> str:
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
    last_error = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV read failed for {path}: {last_error}")


def normalize_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d+)", expand=False).fillna("")
    return s.str.zfill(5)


def to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "-": np.nan, "None": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")


def get_column(cols: List[str], candidates: List[str], contains: bool = False) -> str:
    for cand in candidates:
        for col in cols:
            if contains and cand in col:
                return col
            if (not contains) and cand == col:
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
            and any((c == "소") or ("소" in c) for c in cols)
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


def load_data(root: Path) -> Dict[str, object]:
    paths = discover_inputs(root)

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

    return {
        "paths": paths,
        "shp": shp,
        "emissions": emissions,
        "livestock": livestock,
    }


def preprocess(data: Dict[str, object]) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    shp = data["shp"].copy()
    emissions = data["emissions"].copy()
    livestock = data["livestock"].copy()

    gdf = shp.merge(emissions, on="SIG_CD", how="left")
    gdf = gdf.merge(livestock, on="SIG_CD", how="left")

    raw_area = gdf.geometry.area / 1_000_000
    quality_rows = [
        ("total_regions", len(gdf)),
        ("missing_emissions_count", int(gdf["배출량"].isna().sum())),
        ("missing_emissions_ratio_pct", float(gdf["배출량"].isna().mean() * 100)),
        ("missing_livestock_count", int(gdf["dominant"].isna().sum())),
        ("missing_livestock_ratio_pct", float(gdf["dominant"].isna().mean() * 100)),
        ("area_zero_or_negative_count", int((raw_area <= 0).sum())),
        ("area_min_km2", float(raw_area.min())),
        ("area_p99_km2", float(raw_area.quantile(0.99))),
        ("area_max_km2", float(raw_area.max())),
    ]
    quality_df = pd.DataFrame(quality_rows, columns=["metric", "value"])

    return gdf, quality_df


def calculate_density(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    raw_area_km2 = out.geometry.area / 1_000_000
    out["area_km2"] = raw_area_km2.where(raw_area_km2 > 0, np.nan)

    out["density"] = out["배출량"] / out["area_km2"]
    out["density"] = out["density"].replace([np.inf, -np.inf], np.nan)
    out["log_density"] = np.log1p(out["density"].clip(lower=0))

    cent = out.geometry.centroid
    out["centroid_x"] = cent.x
    out["centroid_y"] = cent.y

    return out


def dominant_group(value: object) -> str:
    if pd.isna(value):
        return "결측"
    txt = str(value)
    if "돼지" in txt or "pig" in txt.lower():
        return "돼지"
    if "소" in txt or "cow" in txt.lower():
        return "소"
    return "기타"


def percentile_hotspot(
    gdf: gpd.GeoDataFrame, percentiles: List[int] | None = None
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    if percentiles is None:
        percentiles = PERCENTILES

    out = gdf.copy()
    density_valid = out["density"].dropna()
    total_emission = out["배출량"].sum(skipna=True)

    rows = []
    for pct in percentiles:
        threshold = float(density_valid.quantile(1 - (pct / 100)))
        col = f"hotspot_p{pct}"
        out[col] = (out["density"] >= threshold).fillna(False)

        hs_mask = out[col]
        nhs_mask = ~hs_mask

        hs_count = int(hs_mask.sum())
        hs_emission = out.loc[hs_mask, "배출량"].sum(skipna=True)
        hs_mean_density = out.loc[hs_mask, "density"].mean()
        nhs_mean_density = out.loc[nhs_mask, "density"].mean()

        hs_share_pct = hs_count / len(out) * 100 if len(out) > 0 else np.nan
        contrib_pct = (hs_emission / total_emission * 100) if total_emission > 0 else np.nan

        rows.append(
            {
                "percentile_top": pct,
                "density_threshold": threshold,
                "hotspot_count": hs_count,
                "hotspot_region_share_pct": hs_share_pct,
                "hotspot_emission_contribution_pct": contrib_pct,
                "hotspot_mean_density": float(hs_mean_density) if pd.notna(hs_mean_density) else np.nan,
                "non_hotspot_mean_density": float(nhs_mean_density) if pd.notna(nhs_mean_density) else np.nan,
                "mean_density_gap": float(hs_mean_density - nhs_mean_density)
                if pd.notna(hs_mean_density) and pd.notna(nhs_mean_density)
                else np.nan,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df["structural_concentration_index"] = (
        summary_df["hotspot_emission_contribution_pct"] / summary_df["hotspot_region_share_pct"]
    )
    return out, summary_df


def sensitivity_analysis(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    show_cols = [
        "percentile_top",
        "hotspot_count",
        "hotspot_emission_contribution_pct",
        "hotspot_mean_density",
        "non_hotspot_mean_density",
        "mean_density_gap",
        "structural_concentration_index",
    ]

    print("\n[Percentile 민감도 분석]")
    print(summary_df[show_cols].round(3).to_string(index=False))

    best_row = summary_df.sort_values("structural_concentration_index", ascending=False).iloc[0]
    best_pct = int(best_row["percentile_top"])
    best_idx = float(best_row["structural_concentration_index"])

    interpretation = (
        f"구조적 집중도(기여율/지역비중) 기준으로 상위 {best_pct}% 기준이 가장 집중 구조를 잘 보여줍니다 "
        f"(index={best_idx:.2f})."
    )
    print("\n[민감도 해석]")
    print(interpretation)

    return summary_df, interpretation


def kmeans_hotspot(
    gdf: gpd.GeoDataFrame,
    k_values: List[int] | None = None,
    default_k: int = DEFAULT_K,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Dict[str, str]]:
    if k_values is None:
        k_values = K_VALUES

    out = gdf.copy()
    summary_rows = []
    feature_sets = {
        "density_only": ["log_density"],
        "density_spatial": ["log_density", "centroid_x", "centroid_y"],
    }

    created_cluster_cols: List[str] = []
    created_hotspot_cols: List[str] = []

    for set_name, features in feature_sets.items():
        for k in k_values:
            cluster_col = f"cluster_{set_name}_k{k}"
            hotspot_col = f"hotspot_kmeans_{set_name}_k{k}"

            out[cluster_col] = pd.NA
            out[hotspot_col] = False

            mask = out[features + ["density", "배출량"]].notna().all(axis=1)
            if int(mask.sum()) < k:
                continue

            x = out.loc[mask, features].to_numpy()
            x_scaled = StandardScaler().fit_transform(x)

            model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
            labels = model.fit_predict(x_scaled)
            out.loc[mask, cluster_col] = labels

            stats = (
                out.loc[mask]
                .groupby(cluster_col, dropna=True)
                .agg(
                    region_count=(cluster_col, "size"),
                    mean_density=("density", "mean"),
                    median_density=("density", "median"),
                    emission_sum=("배출량", "sum"),
                )
                .reset_index()
            )

            hotspot_cluster = int(stats.sort_values("mean_density", ascending=False).iloc[0][cluster_col])
            out.loc[mask, hotspot_col] = out.loc[mask, cluster_col].astype(int) == hotspot_cluster

            total_valid_emission = out.loc[mask, "배출량"].sum()
            for _, row in stats.iterrows():
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
                        "emission_contribution_pct": float(emission_sum / total_valid_emission * 100)
                        if total_valid_emission > 0
                        else np.nan,
                    }
                )

            created_cluster_cols.append(cluster_col)
            created_hotspot_cols.append(hotspot_col)

    default_cluster_col = f"cluster_density_spatial_k{default_k}"
    default_hotspot_col = f"hotspot_kmeans_density_spatial_k{default_k}"

    if default_cluster_col not in out.columns or out[default_cluster_col].notna().sum() == 0:
        if created_cluster_cols:
            default_cluster_col = created_cluster_cols[0]
            default_hotspot_col = created_hotspot_cols[0]
        else:
            raise RuntimeError("K-means failed for all configurations.")

    meta = {
        "default_cluster_col": default_cluster_col,
        "default_hotspot_col": default_hotspot_col,
    }

    return out, pd.DataFrame(summary_rows), meta


def chi_square_test(gdf: gpd.GeoDataFrame, hotspot_col: str = "hotspot_p15") -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    tmp = gdf[[hotspot_col, "dominant_group"]].copy()
    tmp = tmp.dropna(subset=[hotspot_col])
    tmp["hotspot_status"] = np.where(tmp[hotspot_col], "hotspot", "non_hotspot")

    ctab = pd.crosstab(tmp["hotspot_status"], tmp["dominant_group"])

    if ctab.shape[0] < 2 or ctab.shape[1] < 2:
        chi2_val = np.nan
        p_val = np.nan
        dof = 0
        interpretation = "카이제곱 검정을 수행하기에 교차표 차원이 부족합니다."
    else:
        chi2_val, p_val, dof, _ = chi2_contingency(ctab)
        sig = p_val < 0.05
        interpretation = (
            f"p-value={p_val:.4g}로 0.05보다 {'작아' if sig else '크므로'} "
            f"Hotspot/Non-hotspot 간 우세 축종 분포 차이는 {'통계적으로 유의합니다' if sig else '통계적으로 유의하지 않습니다'}."
        )

    print("\n[Hotspot vs Dominant 교차표]")
    print(ctab.to_string())
    print("\n[Chi-square 결과]")
    print(f"chi2 statistic: {chi2_val}")
    print(f"p-value: {p_val}")
    print(interpretation)

    stats_df = pd.DataFrame(
        [
            {
                "chi2_statistic": chi2_val,
                "p_value": p_val,
                "dof": dof,
                "is_significant_p_lt_0_05": bool(p_val < 0.05) if pd.notna(p_val) else False,
                "interpretation": interpretation,
            }
        ]
    )

    crosstab_long = ctab.stack().reset_index()
    crosstab_long.columns = ["hotspot_status", "dominant_group", "count"]

    return stats_df, crosstab_long, interpretation


def save_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    out_png: Path,
    legend_label: str,
    cmap: str = "YlOrRd",
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=MAP_SIZE)
    gdf.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        legend=True,
        vmax=vmax,
        linewidth=0.25,
        edgecolor="#F8F8F8",
        legend_kwds={"label": legend_label, "orientation": "horizontal", "shrink": 0.55, "pad": 0.02},
        missing_kwds={"color": "#E8E8E8", "label": "결측"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_facecolor("#EEF1F4")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.08)
    fig.savefig(out_png, dpi=FIGURE_DPI)
    plt.close(fig)


def save_hotspot_map_emphasized(gdf: gpd.GeoDataFrame, hotspot_col: str, out_png: Path) -> None:
    hs = gdf[gdf[hotspot_col].fillna(False)]
    nhs = gdf[~gdf[hotspot_col].fillna(False)]

    fig, ax = plt.subplots(1, 1, figsize=MAP_SIZE)
    if len(nhs) > 0:
        nhs.plot(ax=ax, color="#D7DDE3", linewidth=0.25, edgecolor="#F3F4F5")
    if len(hs) > 0:
        hs.plot(ax=ax, color="#D1492F", linewidth=0.3, edgecolor="#F3F4F5")
        hs.boundary.plot(ax=ax, color="#8E1B10", linewidth=1.8)
        hs.plot(ax=ax, facecolor="none", edgecolor="#8E1B10", linewidth=0.0, hatch="///")

    handles = [
        Patch(facecolor="#D1492F", edgecolor="#8E1B10", label="Hotspot"),
        Patch(facecolor="#D7DDE3", edgecolor="#444444", label="Non-hotspot"),
    ]
    ax.legend(handles=handles, loc="lower left")
    ax.set_title("상위 15% Hotspot 강조 지도")
    ax.set_axis_off()
    ax.set_facecolor("#EEF1F4")
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


def save_density_hist(gdf: gpd.GeoDataFrame, out_png: Path, p15_threshold: float) -> None:
    vals = gdf["density"].dropna()
    vals = vals[vals > 0]
    if len(vals) == 0:
        return

    bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 36)
    median = float(np.median(vals))

    fig, ax = plt.subplots(figsize=CHART_SIZE)
    ax.hist(vals, bins=bins, color="#2C7FB8", edgecolor="white", alpha=0.92)
    ax.axvline(median, color="#1B4F72", linestyle="--", linewidth=1.5, label=f"중앙값 {median:,.0f}")
    ax.axvline(p15_threshold, color="#C0392B", linestyle="-.", linewidth=1.5, label=f"상위 15% 경계 {p15_threshold:,.0f}")
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


def make_maps(gdf: gpd.GeoDataFrame, out_fig_dir: Path, kmeans_meta: Dict[str, str], p15_threshold: float) -> None:
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    save_choropleth(gdf, "배출량", "시군구 NH3 배출량", out_fig_dir / "map_emissions_kgyr.png", "배출량 (kg/yr)")

    save_choropleth(
        gdf,
        "density",
        "시군구 NH3 배출밀도",
        out_fig_dir / "map_density_kgkm2yr.png",
        "배출밀도 (kg/km²/yr)",
    )

    density_q95 = float(gdf["density"].dropna().quantile(0.95))
    save_choropleth(
        gdf,
        "density",
        "시군구 NH3 배출밀도 (95% Clip)",
        out_fig_dir / "map_density_clip95.png",
        "배출밀도 (kg/km²/yr)",
        vmax=density_q95,
    )

    max_density = float(gdf["density"].max())
    clipped_count = int((gdf["density"] > density_q95).sum())
    print("\n[배출밀도 지도 비교]")
    print(
        f"Full scale은 최대값 {max_density:,.1f}까지 반영하여 상위 극단값의 색 대비가 강합니다. "
        f"Clip95는 상위 5%({clipped_count}개 지역)를 {density_q95:,.1f}로 제한해 중간 구간 대비를 더 잘 보여줍니다."
    )

    save_hotspot_map(gdf, "hotspot_p15", "상위 15% Hotspot", out_fig_dir / "map_hotspot_percentile_p15.png")
    save_hotspot_map(
        gdf,
        kmeans_meta["default_hotspot_col"],
        "K-means Hotspot",
        out_fig_dir / "map_hotspot_kmeans_default.png",
    )

    save_hotspot_map_emphasized(gdf, "hotspot_p15", out_fig_dir / "map_hotspot_emphasized.png")

    save_dominant_map(gdf, out_fig_dir / "map_dominant_pig_vs_cow.png")
    save_hotspot_dominant_map(gdf, "hotspot_p15", out_fig_dir / "map_hotspot_dominant_pig_vs_cow.png")
    save_density_hist(gdf, out_fig_dir / "hist_density_logscale.png", p15_threshold)
    save_dominant_bar(gdf, "hotspot_p15", out_fig_dir / "bar_dominant_hotspot_vs_nonhotspot.png")




def check_spatial_environment() -> None:
    """Print environment check messages for spatial ESDA dependencies."""
    print("\n[Spatial Environment]")
    try:
        import libpysal  # type: ignore
        print(f"- libpysal: {libpysal.__version__}")
    except Exception as exc:
        print(f"- libpysal: unavailable ({exc})")
    try:
        import esda  # type: ignore
        print(f"- esda: {esda.__version__}")
    except Exception as exc:
        print(f"- esda: unavailable ({exc})")


def compute_spatial_weights(gdf: gpd.GeoDataFrame, method: str = "queen", silence_warnings: bool = False):
    """
    Create contiguity weights from geometry.
    - method: queen (default) or rook
    - row-standardized transform is applied
    - returns cleaned gdf, weight object, and islands list
    """
    work = gdf.copy()
    n_before = len(work)

    work = work[work.geometry.notnull()].copy()
    null_removed = n_before - len(work)

    invalid_removed = 0
    if len(work) > 0:
        valid_mask = work.is_valid
        invalid_removed = int((~valid_mask).sum())
        work = work[valid_mask].copy()

    method_lower = method.lower()
    if method_lower == "queen":
        w = Queen.from_dataframe(work, use_index=True, silence_warnings=True)
    elif method_lower == "rook":
        w = Rook.from_dataframe(work, use_index=True, silence_warnings=True)
    else:
        raise ValueError(f"Unsupported weights method: {method}. Use queen or rook.")

    w.transform = "R"
    islands = list(w.islands)

    print("\n[Spatial Weights]")
    print(f"- method: {method_lower}")
    print(f"- geometry null removed: {null_removed}")
    print(f"- geometry invalid removed: {invalid_removed}")
    print(f"- islands count: {len(islands)}")

    if islands and not silence_warnings:
        print(f"- islands index preview: {islands[:10]}")
        print("  warning: islands (no-neighbor polygons) are included by default.")

    return work, w, islands


def _align_y_and_w(gdf: gpd.GeoDataFrame, value_col: str, w):
    """Drop missing y and subset weights to the same index order."""
    if value_col not in gdf.columns:
        raise KeyError(f"Column not found: {value_col}")

    valid = gdf[value_col].notna()
    g = gdf.loc[valid].copy()
    dropped = int((~valid).sum())

    if len(g) == 0:
        raise ValueError(f"No valid rows for spatial variable: {value_col}")

    if dropped > 0:
        w_use = w_subset(w, list(g.index))
    else:
        w_use = w

    w_use.transform = "R"
    return g, w_use, dropped


def global_moran(gdf: gpd.GeoDataFrame, value_col: str, w, permutations: int = 999):
    """Compute global Moran's I for one variable."""
    g, w_use, dropped = _align_y_and_w(gdf, value_col, w)
    y = g[value_col].to_numpy()

    np.random.seed(RANDOM_STATE)
    moran = Moran(y, w_use, permutations=permutations)

    result = {
        "value_col": value_col,
        "n": len(g),
        "dropped_na_count": dropped,
        "permutations": permutations,
        "moran_i": float(moran.I),
        "expected_i": float(moran.EI),
        "z_score": float(moran.z_sim),
        "p_value": float(moran.p_sim),
    }

    print(f"\n[Global Moran] {value_col}: I={result['moran_i']:.4f}, p={result['p_value']:.4g}")
    return result, g, w_use, moran


def local_moran(gdf: gpd.GeoDataFrame, value_col: str, w, permutations: int = 999, alpha: float = 0.05):
    """Compute Local Moran (LISA) and assign cluster labels."""
    g, w_use, dropped = _align_y_and_w(gdf, value_col, w)
    y = g[value_col].to_numpy()

    np.random.seed(RANDOM_STATE)
    with np.errstate(divide="ignore", invalid="ignore"):
        li = Moran_Local(y, w_use, permutations=permutations, seed=RANDOM_STATE)

    sig = li.p_sim < alpha
    q = li.q

    cluster = np.full(len(g), "Not significant", dtype=object)
    cluster[sig & (q == 1)] = "HH"
    cluster[sig & (q == 2)] = "LH"
    cluster[sig & (q == 3)] = "LL"
    cluster[sig & (q == 4)] = "HL"

    ccol = f"lisa_cluster_{value_col}"
    sig_col = f"lisa_sig_{value_col}"
    hh_col = f"lisa_hh_{value_col}"

    g[f"local_I_{value_col}"] = li.Is
    g[f"local_p_sim_{value_col}"] = li.p_sim
    g[f"local_q_{value_col}"] = li.q
    g[sig_col] = sig
    g[ccol] = cluster
    g[hh_col] = sig & (q == 1)

    sig_count = int(sig.sum())
    hh_count = int(g[hh_col].sum())

    print(f"\n[Local Moran] {value_col}: valid={len(g)}, dropped_na={dropped}, sig={sig_count}, HH={hh_count}")
    return g, {"sig_count": sig_count, "hh_count": hh_count, "cluster_col": ccol, "hh_col": hh_col}, w_use, li


def make_lisa_cluster_map(gdf: gpd.GeoDataFrame, cluster_col: str, outpath: Path, title: str) -> None:
    """Draw LISA cluster map with standard ESDA palette."""
    palette = {
        "HH": "#d7191c",
        "LL": "#2c7bb6",
        "LH": "#abd9e9",
        "HL": "#fdae61",
        "Not significant": "#d9d9d9",
    }
    order = ["HH", "LL", "LH", "HL", "Not significant"]

    fig, ax = plt.subplots(1, 1, figsize=MAP_SIZE)
    for label in order:
        sub = gdf[gdf[cluster_col] == label]
        if len(sub) > 0:
            sub.plot(ax=ax, color=palette[label], linewidth=0.25, edgecolor="#F8F8F8")

    handles = [Patch(facecolor=palette[k], edgecolor="black", label=k) for k in order if (gdf[cluster_col] == k).any()]
    ax.legend(handles=handles, loc="lower left")
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_facecolor("#EEF1F4")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.08)
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)


def compare_hotspots(gdf: pd.DataFrame, p15_col: str, lisa_hh_col: str):
    """Create 2x2 cross comparison between percentile hotspot and LISA HH."""
    tmp = gdf[[p15_col, lisa_hh_col]].copy()
    tmp[p15_col] = tmp[p15_col].fillna(False).astype(bool)
    tmp[lisa_hh_col] = tmp[lisa_hh_col].fillna(False).astype(bool)

    ctab = pd.crosstab(tmp[p15_col], tmp[lisa_hh_col]).reindex(index=[False, True], columns=[False, True], fill_value=0)

    overlap = int(ctab.loc[True, True])
    p15_total = int(ctab.loc[True].sum())
    hh_total = int(ctab[True].sum())

    p15_in_hh = (overlap / p15_total * 100) if p15_total > 0 else np.nan
    hh_in_p15 = (overlap / hh_total * 100) if hh_total > 0 else np.nan

    compare_df = (
        ctab.stack()
        .rename("count")
        .reset_index()
        .rename(columns={p15_col: "p15_hotspot", lisa_hh_col: "lisa_hh"})
    )

    metrics = {
        "overlap_count": overlap,
        "p15_total": p15_total,
        "hh_total": hh_total,
        "p15_in_hh_pct": p15_in_hh,
        "hh_in_p15_pct": hh_in_p15,
    }
    return compare_df, metrics


def chi_square_dominant(gdf: pd.DataFrame, hotspot_col: str, dominant_col: str):
    """Chi-square test for dominant distribution by hotspot grouping."""
    tmp = gdf[[hotspot_col, dominant_col]].copy()
    tmp = tmp.dropna(subset=[dominant_col])
    tmp["group"] = np.where(tmp[hotspot_col].fillna(False), "LISA_HH", "Non_HH")

    ctab = pd.crosstab(tmp["group"], tmp[dominant_col])
    if ctab.shape[0] < 2 or ctab.shape[1] < 2:
        chi2_val = np.nan
        p_val = np.nan
        dof = 0
        interpretation = "???? ??? ????? ??? ??? ?????."
    else:
        chi2_val, p_val, dof, _ = chi2_contingency(ctab)
        interpretation = (
            f"p-value={p_val:.4g}; dominant distribution differs by LISA_HH status: "
            f"{'significant' if p_val < 0.05 else 'not significant'}."
        )

    stats_df = pd.DataFrame(
        [
            {
                "chi2_statistic": chi2_val,
                "p_value": p_val,
                "dof": dof,
                "is_significant_p_lt_0_05": bool(p_val < 0.05) if pd.notna(p_val) else False,
                "interpretation": interpretation,
            }
        ]
    )
    ctab_long = ctab.stack().reset_index()
    ctab_long.columns = ["group", "dominant_group", "count"]
    return stats_df, ctab_long, interpretation


def save_lisa_dominant_bar(gdf: pd.DataFrame, hotspot_col: str, dominant_col: str, out_png: Path) -> None:
    tmp = gdf[[hotspot_col, dominant_col]].copy()
    tmp["status"] = np.where(tmp[hotspot_col].fillna(False), "LISA_HH", "Non_HH")
    dist = pd.crosstab(tmp["status"], tmp[dominant_col], normalize="index") * 100

    status_order = ["LISA_HH", "Non_HH"]
    col_order = [c for c in ["??", "?", "??", "??"] if c in dist.columns]
    dist = dist.reindex(status_order).fillna(0.0)
    if col_order:
        dist = dist[col_order]

    fig, ax = plt.subplots(figsize=CHART_SIZE)
    dist.plot(kind="bar", stacked=True, ax=ax, width=0.48, color=["#D95C8A", "#6E4A35", "#54A24B", "#9E9E9E"])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_ylabel("?? (%)")
    ax.set_xlabel("")
    ax.set_title("LISA HH vs Non-HH ?? ?? ??")
    ax.set_xticklabels([str(x.get_text()) for x in ax.get_xticklabels()], rotation=0, ha="center")
    ax.legend(title="dominant", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(left=0.12, right=0.82, top=0.88, bottom=0.15)
    fig.savefig(out_png, dpi=FIGURE_DPI)
    plt.close(fig)


def summarize_lisa_hh_locations(gdf_lisa_density: gpd.GeoDataFrame, hh_col: str, top_n: int = 3) -> str:
    hh = gdf_lisa_density[gdf_lisa_density[hh_col]].copy()
    if len(hh) == 0:
        return "No LISA HH regions detected for location summary."

    hh["sido_code"] = hh["SIG_CD"].astype(str).str[:2]
    hh["sido_name"] = hh["sido_code"].map(SPATIAL_SIDO_MAP).fillna("??")

    top = hh["sido_name"].value_counts().head(top_n)
    parts = [f"{name}({cnt})" for name, cnt in top.items()]
    return "LISA HH is concentrated in: " + ", ".join(parts)


def run_spatial_autocorrelation(gdf: gpd.GeoDataFrame, root: Path, weights_method: str = "queen", permutations: int = 999, alpha: float = 0.05, drop_islands: bool = False):
    """Run ESDA workflow and save outputs in outputs_v2 without touching existing outputs."""
    check_spatial_environment()

    out_v2 = root / "outputs_v2"
    out_fig = out_v2 / "figures"
    out_tbl = out_v2 / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tbl.mkdir(parents=True, exist_ok=True)

    gdf_spatial, w, islands = compute_spatial_weights(gdf, method=weights_method, silence_warnings=False)

    if drop_islands and islands:
        gdf_spatial = gdf_spatial.drop(index=islands).copy()
        print(f"- drop_islands=True applied: removed {len(islands)} regions and recomputed weights")
        gdf_spatial, w, islands = compute_spatial_weights(gdf_spatial, method=weights_method, silence_warnings=True)
        print(f"- islands count after drop: {len(islands)}")

    g_res_density, _, _, _ = global_moran(gdf_spatial, "density", w, permutations=permutations)
    g_res_log, _, _, _ = global_moran(gdf_spatial, "log_density", w, permutations=permutations)

    global_df = pd.DataFrame([g_res_density, g_res_log])
    global_df.to_csv(out_tbl / "moran_global_summary.csv", index=False, encoding="utf-8-sig")

    lisa_density, info_density, _, _ = local_moran(
        gdf_spatial, "density", w, permutations=permutations, alpha=alpha
    )
    lisa_log, info_log, _, _ = local_moran(
        gdf_spatial, "log_density", w, permutations=permutations, alpha=alpha
    )

    make_lisa_cluster_map(
        lisa_density,
        info_density["cluster_col"],
        out_fig / "map_lisa_clusters_density.png",
        "LISA Cluster Map (density)",
    )
    make_lisa_cluster_map(
        lisa_log,
        info_log["cluster_col"],
        out_fig / "map_lisa_clusters_logdensity.png",
        "LISA Cluster Map (log_density)",
    )

    density_cols = [
        "SIG_CD", "SIG_KOR_NM", "density", "log_density",
        "local_I_density", "local_p_sim_density", "local_q_density",
        "lisa_sig_density", "lisa_cluster_density", "lisa_hh_density",
    ]
    log_cols = [
        "SIG_CD", "SIG_KOR_NM", "density", "log_density",
        "local_I_log_density", "local_p_sim_log_density", "local_q_log_density",
        "lisa_sig_log_density", "lisa_cluster_log_density", "lisa_hh_log_density",
    ]

    lisa_density.loc[:, [c for c in density_cols if c in lisa_density.columns]].to_csv(
        out_tbl / "lisa_local_results_density.csv", index=False, encoding="utf-8-sig"
    )
    lisa_log.loc[:, [c for c in log_cols if c in lisa_log.columns]].to_csv(
        out_tbl / "lisa_local_results_logdensity.csv", index=False, encoding="utf-8-sig"
    )

    compare_base = gdf[["SIG_CD", "hotspot_p15", "dominant_group"]].copy()
    compare_base = compare_base.merge(
        lisa_density[["SIG_CD", "lisa_hh_density", "lisa_cluster_density"]],
        on="SIG_CD",
        how="left",
    )
    compare_base["lisa_hh_density"] = compare_base["lisa_hh_density"].map(lambda v: bool(v) if pd.notna(v) else False)

    compare_df, overlap = compare_hotspots(compare_base, "hotspot_p15", "lisa_hh_density")
    compare_df.to_csv(out_tbl / "compare_p15_vs_lisa_hh.csv", index=False, encoding="utf-8-sig")

    print("\n[Hotspot Overlap: p15 vs LISA HH]")
    print(f"- share of p15 that is LISA HH: {overlap['p15_in_hh_pct']:.1f}%")
    print(f"- share of LISA HH that is p15: {overlap['hh_in_p15_pct']:.1f}%")

    global_sig_sentence = (
        "NH3 density shows statistically significant spatial clustering."
        if g_res_density["p_value"] < 0.05
        else "NH3 density does not show statistically significant spatial clustering."
    )
    location_sentence = summarize_lisa_hh_locations(lisa_density, "lisa_hh_density", top_n=3)
    print("\n[Poster-ready sentences]")
    print(global_sig_sentence)
    print(location_sentence)

    chi_stats, chi_ctab, chi_note = chi_square_dominant(compare_base, "lisa_hh_density", "dominant_group")

    chi_stats_exp = chi_stats.copy()
    chi_stats_exp["record_type"] = "chi_square"
    chi_stats_exp["group"] = pd.NA
    chi_stats_exp["dominant_group"] = pd.NA
    chi_stats_exp["count"] = pd.NA

    chi_ctab_exp = chi_ctab.copy()
    chi_ctab_exp["record_type"] = "contingency"
    chi_ctab_exp["chi2_statistic"] = np.nan
    chi_ctab_exp["p_value"] = np.nan
    chi_ctab_exp["dof"] = np.nan
    chi_ctab_exp["is_significant_p_lt_0_05"] = np.nan
    chi_ctab_exp["interpretation"] = pd.NA

    chi_cols = [
        "record_type", "chi2_statistic", "p_value", "dof", "is_significant_p_lt_0_05", "interpretation",
        "group", "dominant_group", "count",
    ]
    pd.concat([chi_stats_exp[chi_cols], chi_ctab_exp[chi_cols]], ignore_index=True).to_csv(
        out_tbl / "chi_square_lisa_hh_dominant.csv", index=False, encoding="utf-8-sig"
    )

    save_lisa_dominant_bar(compare_base, "lisa_hh_density", "dominant_group", out_fig / "bar_dominant_lisa_hh_vs_non.png")

    print("\n[Spatial Summary Logs]")
    print(f"1) weights method={weights_method}, islands={len(islands)}")
    print(f"2) global Moran density I={g_res_density['moran_i']:.4f}, p={g_res_density['p_value']:.4g}")
    print(f"   global Moran log_density I={g_res_log['moran_i']:.4f}, p={g_res_log['p_value']:.4g}")
    print(f"3) local Moran significant count (density)={info_density['sig_count']}, (log_density)={info_log['sig_count']}")
    print(f"4) LISA HH count (density)={info_density['hh_count']}")
    print(f"5) overlap metrics: p15->HH={overlap['p15_in_hh_pct']:.1f}%, HH->p15={overlap['hh_in_p15_pct']:.1f}%")
    print(f"6) dominant chi-square p-value={float(chi_stats.iloc[0]['p_value']):.4g}")

    return {
        "global_density": g_res_density,
        "global_log_density": g_res_log,
        "overlap": overlap,
        "chi_square_note": chi_note,
        "global_sentence": global_sig_sentence,
        "location_sentence": location_sentence,
        "outputs_v2": out_v2,
    }
def export_results(
    gdf: gpd.GeoDataFrame,
    quality_df: pd.DataFrame,
    percentile_df: pd.DataFrame,
    kmeans_df: pd.DataFrame,
    chi_stats_df: pd.DataFrame,
    chi_ctab_df: pd.DataFrame,
    out_tbl_dir: Path,
    out_dir: Path,
    kmeans_meta: Dict[str, str],
    kpi_lines: List[str],
) -> None:
    out_tbl_dir.mkdir(parents=True, exist_ok=True)

    quality_df.to_csv(out_tbl_dir / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    percentile_df.to_csv(out_tbl_dir / "percentile_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    percentile_df.to_csv(out_tbl_dir / "percentile_hotspot_contribution.csv", index=False, encoding="utf-8-sig")
    kmeans_df.to_csv(out_tbl_dir / "kmeans_cluster_summary.csv", index=False, encoding="utf-8-sig")

    chi_stats_expanded = chi_stats_df.copy()
    chi_stats_expanded["record_type"] = "chi_square"
    chi_stats_expanded["hotspot_status"] = pd.NA
    chi_stats_expanded["dominant_group"] = pd.NA
    chi_stats_expanded["count"] = pd.NA

    chi_ctab_expanded = chi_ctab_df.copy()
    chi_ctab_expanded["record_type"] = "contingency"
    chi_ctab_expanded["chi2_statistic"] = np.nan
    chi_ctab_expanded["p_value"] = np.nan
    chi_ctab_expanded["dof"] = np.nan
    chi_ctab_expanded["is_significant_p_lt_0_05"] = np.nan
    chi_ctab_expanded["interpretation"] = pd.NA

    chi_cols = [
        "record_type",
        "chi2_statistic",
        "p_value",
        "dof",
        "is_significant_p_lt_0_05",
        "interpretation",
        "hotspot_status",
        "dominant_group",
        "count",
    ]
    chi_out = pd.concat([chi_stats_expanded[chi_cols], chi_ctab_expanded[chi_cols]], ignore_index=True)
    chi_out.to_csv(out_tbl_dir / "chi_square_hotspot_dominant.csv", index=False, encoding="utf-8-sig")

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
        out_tbl_dir / "region_level_results.csv", index=False, encoding="utf-8-sig"
    )

    (out_dir / "kpi_summary.txt").write_text("\n".join(kpi_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Korea SIG NH3 hotspot analysis pipeline")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root path")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "outputs", help="Output directory")
    parser.add_argument("--weights-method", choices=["queen", "rook"], default="queen", help="Spatial weights method")
    parser.add_argument("--moran-permutations", type=int, default=999, help="Permutation count for Moran tests")
    parser.add_argument("--lisa-alpha", type=float, default=0.05, help="Significance alpha for LISA")
    parser.add_argument("--drop-islands", action="store_true", help="Drop islands before Moran calculation")
    args = parser.parse_args()

    selected_font = configure_plot_style()

    data = load_data(args.root)
    gdf, quality_df = preprocess(data)
    gdf = calculate_density(gdf)
    gdf["dominant_group"] = gdf["dominant"].map(dominant_group)

    print("[INPUTS]")
    for key, path in data["paths"].items():
        print(f"- {key}: {path}")
    print(f"- plot_font: {selected_font}")

    gdf, percentile_df = percentile_hotspot(gdf, PERCENTILES)
    percentile_df, sensitivity_note = sensitivity_analysis(percentile_df)

    gdf, kmeans_df, kmeans_meta = kmeans_hotspot(gdf, K_VALUES, DEFAULT_K)

    chi_stats_df, chi_ctab_df, chi_note = chi_square_test(gdf, "hotspot_p15")

    hs_mask = gdf["hotspot_p15"].fillna(False)
    nhs_mask = ~hs_mask

    total_emission = gdf["배출량"].sum(skipna=True)
    hs_emission = gdf.loc[hs_mask, "배출량"].sum(skipna=True)
    hs_emission_pct = float(hs_emission / total_emission * 100) if total_emission > 0 else np.nan

    hs_mean_density = float(gdf.loc[hs_mask, "density"].mean())
    nhs_mean_density = float(gdf.loc[nhs_mask, "density"].mean())
    density_ratio = hs_mean_density / nhs_mean_density if nhs_mean_density > 0 else np.nan

    pig_hotspot_pct = float((gdf.loc[hs_mask, "dominant_group"] == "돼지").mean() * 100) if hs_mask.sum() > 0 else np.nan

    kpi_1 = f"KPI 1: 상위 15 percent 지역이 전체 NH3 배출의 {hs_emission_pct:.1f} percent 차지"
    kpi_2 = f"KPI 2: Hotspot 평균 밀도는 Non-hotspot 대비 {density_ratio:.2f} 배 높음"
    kpi_3 = f"KPI 3: Hotspot 중 {pig_hotspot_pct:.1f} percent가 돼지 dominant"

    print("\n[KPI]")
    print(kpi_1)
    print(kpi_2)
    print(kpi_3)

    p15_row = percentile_df[percentile_df["percentile_top"] == 15].iloc[0]
    p15_threshold = float(p15_row["density_threshold"])
    out_fig = args.output / "figures"
    out_tbl = args.output / "tables"
    make_maps(gdf, out_fig, kmeans_meta, p15_threshold)

    spatial_outputs = run_spatial_autocorrelation(
        gdf=gdf,
        root=args.root,
        weights_method=args.weights_method,
        permutations=args.moran_permutations,
        alpha=args.lisa_alpha,
        drop_islands=args.drop_islands,
    )

    export_results(
        gdf=gdf,
        quality_df=quality_df,
        percentile_df=percentile_df,
        kmeans_df=kmeans_df,
        chi_stats_df=chi_stats_df,
        chi_ctab_df=chi_ctab_df,
        out_tbl_dir=out_tbl,
        out_dir=args.output,
        kmeans_meta=kmeans_meta,
        kpi_lines=[kpi_1, kpi_2, kpi_3, "", f"민감도 해석: {sensitivity_note}", f"카이제곱 해석: {chi_note}"],
    )

    print("\n[OUTPUTS]")
    print(f"- figures: {out_fig}")
    print(f"- tables: {out_tbl}")
    print(f"- kpi: {args.output / 'kpi_summary.txt'}")
    print(f"- outputs_v2: {spatial_outputs['outputs_v2']}")


if __name__ == "__main__":
    main()
