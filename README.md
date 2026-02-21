# Korea NH3 Spatial Hotspot Analysis

대한민국 시군구 단위 NH3 배출 데이터를 기반으로,
핫스팟 탐지(Percentile, K-means)와 공간자기상관(ESDA: Global/Local Moran, LISA)까지 수행하는 분석 파이프라인입니다.

이 저장소는 실무/발표 목적에 맞게 다음 작업들을 반영했습니다.
- 시군구 경계 + 배출량 + 축종(dominant) 자동 병합
- 면적 보정 배출밀도(`density`) 및 로그 밀도(`log_density`) 계산
- 상위 분위 기반 핫스팟(10/15/20%) + K-means 핫스팟
- 통계 검정(chi-square) 및 KPI 자동 출력
- Moran's I / LISA 기반 공간 군집 분석 및 비교 리포트
- 발표용 지도/그래프/테이블 자동 생성

## Project Structure
- `nh3_hotspot_pipeline.py`: 전체 엔드투엔드 파이프라인
- `outputs/`: 기존 분석 결과(핫스팟/지도/표/KPI)
- `outputs_v2/`: ESDA 확장 결과(Moran/LISA/비교 분석)

## Main Pipeline (요약)
1. 데이터 로딩/정합성 확인 (`load_data`, `preprocess`)
2. 면적/배출밀도/로그밀도 계산 (`calculate_density`)
3. Percentile 핫스팟 및 민감도 분석 (`percentile_hotspot`, `sensitivity_analysis`)
4. K-means 핫스팟 분석 (`kmeans_hotspot`)
5. 통계 검정 (`chi_square_test`)
6. 지도/그래프 생성 (`make_maps`)
7. ESDA 공간자기상관 분석 (`run_spatial_autocorrelation`)
   - Queen/Rook contiguity weights (기본 Queen)
   - Global Moran's I (`density`, `log_density`)
   - Local Moran (LISA) cluster map
   - p15 vs LISA HH overlap 비교
   - LISA HH vs dominant chi-square
8. 결과 저장 (`export_results`)

## Run
기본 실행:
```bash
python nh3_hotspot_pipeline.py
```

공간가중치/퍼뮤테이션 옵션 포함 실행:
```bash
python nh3_hotspot_pipeline.py --weights-method queen --moran-permutations 999
```

선택 옵션:
- `--weights-method {queen,rook}`
- `--moran-permutations` (기본 999)
- `--lisa-alpha` (기본 0.05)
- `--drop-islands` (기본 false)

## Key Outputs
### Existing outputs (`outputs/`)
- `outputs/figures/map_emissions_kgyr.png`
- `outputs/figures/map_density_kgkm2yr.png`
- `outputs/figures/map_density_clip95.png`
- `outputs/figures/map_hotspot_percentile_p15.png`
- `outputs/figures/map_hotspot_emphasized.png`
- `outputs/tables/percentile_sensitivity_summary.csv`
- `outputs/tables/chi_square_hotspot_dominant.csv`
- `outputs/kpi_summary.txt`

### ESDA outputs (`outputs_v2/`)
- `outputs_v2/tables/moran_global_summary.csv`
- `outputs_v2/tables/lisa_local_results_density.csv`
- `outputs_v2/tables/lisa_local_results_logdensity.csv`
- `outputs_v2/tables/compare_p15_vs_lisa_hh.csv`
- `outputs_v2/tables/chi_square_lisa_hh_dominant.csv`
- `outputs_v2/figures/map_lisa_clusters_density.png`
- `outputs_v2/figures/map_lisa_clusters_logdensity.png`
- `outputs_v2/figures/bar_dominant_lisa_hh_vs_non.png`

## Dependencies
필수 패키지(주요):
- `geopandas`, `pandas`, `numpy`, `matplotlib`
- `scikit-learn`, `scipy`
- `libpysal`, `esda`

## Notes
- 기존 `outputs` 폴더는 유지하고, ESDA 확장 산출물은 `outputs_v2`에 별도 저장합니다.
- 데이터 파일 경로는 프로젝트 내부에서 자동 탐색되며, 코드 키(`SIG_CD`, `CD`)는 문자열 zfill(5)로 정규화됩니다.
