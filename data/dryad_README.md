# UAV Imagery and Model Data for Southern Leaf Blight (SLB) Severity Phenotyping in Maize

## Description

This dataset contains unmanned aerial vehicle (UAV) raw imagery, plot-level image
crops, and trained model checkpoints supporting a deep learning pipeline for
quantitative phenotyping of Southern Leaf Blight (SLB) disease severity in maize.
Data were collected across three growing seasons (2023–2025) from two fields at the same
research site per year. The pipeline uses transfer learning (EVA-02-B) applied to UAV
RGB imagery to estimate disease severity ratings.

This dataset supports the manuscript (in preparation, target journal:
*The Plant Phenome Journal*) on UAV-based deep learning for SLB severity estimation.

Code, training scripts, and processed/intermediate data products (e.g. CV split
definitions, orthomosaics, canopy height rasters) are not included in this
dataset; the analysis pipeline is maintained separately in the associated
project GitHub repository.

## Dataset structure

```
DJI_<flightdate>_<field>-<year>.tar.gz           28 archives — raw UAV flight imagery + GNSS logs
plot_images_<field>_<year>.tar.gz                 6 archives — cropped per-plot image tiles (JPG)
<field>_plot_outline.tar.gz                       6 archives — plot boundary shapefiles
eva02_base_cv0_fold_<N>_checkpoint_best.pt        3 files   — best-epoch model weights per CV fold
full_dataset.csv                                  1 file    — full image/severity-score metadata table
flight_covariates.csv                             1 file    — per-flight environmental/acquisition covariates
```

Total dataset size: ~867 GB across 45 files. The complete file list, with
exact filenames and sizes, is given in each subsection below.

### `DJI_<flightdate>_<field>-<year>.tar.gz` (27 archives)

Each archive corresponds to one UAV flight (a single field on a single date) and
contains:
- Raw `.JPG` images captured during that flight
- Associated GNSS post-processed kinematic (PPK) correction files: `.nav`,
  `.obs`, `.bin` (raw GNSS observation/navigation logs), `.MRK` (image timestamp/
  position marks)

Archive naming follows the source flight folder name:
`DJI_<YYYYMMDD>_<FIELD>-<YEAR>.tar.gz`

Flights included, by field and year, with the exact archive filename and size:

| Year | Plot | Flight date | Filename | Size |
|---|---|---|---|---|
| 2023 | I3B | 2023-07-13 | `DJI_20230713_I3B-2023.tar.gz` | 34G |
| 2023 | I3B | 2023-07-20 | `DJI_20230720_I3B-2023.tar.gz` | 32G |
| 2023 | I3B | 2023-07-27 | `DJI_20230727_I3B-2023.tar.gz` | 31G |
| 2023 | G3  | 2023-07-20 | `DJI_20230720_G3-2023.tar.gz` | 37G |
| 2023 | G3  | 2023-07-27 | `DJI_20230727_G3-2023.tar.gz` | 47G |
| 2024 | B7  | 2024-07-03 | `DJI_20240703_B7-2024.tar.gz` | 21G |
| 2024 | B7  | 2024-07-10 | `DJI_20240710_B7-2024.tar.gz` | 22G |
| 2024 | B7  | 2024-07-16 | `DJI_20240716_B7-2024.tar.gz` | 23G |
| 2024 | B7  | 2024-07-27 | `DJI_20240727_B7-2024.tar.gz` | 25G |
| 2024 | C6B | 2024-07-03 | `DJI_20240703_C6B-2024.tar.gz` | 51G |
| 2024 | C6B | 2024-07-10 | `DJI_20240710_C6B-2024.tar.gz` | 51G |
| 2024 | C6B | 2024-07-18 | `DJI_20240718_C6B-2024.tar.gz` | 54G |
| 2024 | C6B | 2024-07-26 | `DJI_20240726_C6B-2024.tar.gz` | 51G |
| 2024 | C6B | 2024-08-02 | `DJI_20240802_C6B-2024.tar.gz` | 14G |
| 2025 | C10 | 2025-06-28 | `DJI_20250628_C10-2025.tar.gz` | 31G |
| 2025 | C10 | 2025-07-04 | `DJI_20250704_C10-2025.tar.gz` | 24G |
| 2025 | C10 | 2025-07-07 | `DJI_20250707_C10-2025.tar.gz` | 27G |
| 2025 | C10 | 2025-07-11 | `DJI_20250711_C10-2025.tar.gz` | 29G |
| 2025 | C10 | 2025-07-14 | `DJI_20250714_C10-2025.tar.gz` | 27G |
| 2025 | C10 | 2025-07-18 | `DJI_20250718_C10-2025.tar.gz` | 29G |
| 2025 | C10 | 2025-07-21 | `DJI_20250721_C10-2025.tar.gz` | 26G |
| 2025 | C7B | 2025-07-01 | `DJI_20250701_C7B-2025.tar.gz` | 23G |
| 2025 | C7B | 2025-07-04 | `DJI_20250704_C7B-2025.tar.gz` | 23G |
| 2025 | C7B | 2025-07-07 | `DJI_20250707_C7B-2025.tar.gz` | 24G |
| 2025 | C7B | 2025-07-10 | `DJI_20250710_C7B-2025.tar.gz` | 23G |
| 2025 | C7B | 2025-07-12 | `DJI_20250712_C7B-2025.tar.gz` | 22G |
| 2025 | C7B | 2025-07-18 | `DJI_20250718_C7B-2025.tar.gz` | 22G |

Imagery was captured with a DJI Zenmuse P1 35mm camera; flight operations were
conducted under FAA Part 107 certification.

### `plot_images_<field>_<year>.tar.gz` (6 archives)

Cropped, plot-level JPG image tiles extracted for disease scoring and model
training/evaluation, grouped by plot and year (26,071 images total). Filenames
encode acquisition date, field, and plot number:
`<YYYYMMDD>_<FIELD>_<PLOT>.jpg`

Note: the `B7` plot is labeled `B7A` in these filenames; this refers to the
same field plot as `B7` in the raw flights archives.

| Filename | Plot | Year | Size |
|---|---|---|---|
| `plot_images_G3_2023.tar.gz` | G3 | 2023 | 770M |
| `plot_images_I3B_2023.tar.gz` | I3B | 2023 | 1013M |
| `plot_images_B7_2024.tar.gz` | B7A | 2024 | 1.7G |
| `plot_images_C6B_2024.tar.gz` | C6B | 2024 | 4.4G |
| `plot_images_C10_2025.tar.gz` | C10 | 2025 | 5.2G |
| `plot_images_C7B_2025.tar.gz` | C7B | 2025 | 5.5G |

### `<field>_plot_outline.tar.gz` (6 archives)

Plot boundary shapefiles delineating the extent of each plot (B7A, C10,
C6B, C7B, G3, I3B), used to crop/align processed flight imagery to plot-level
tiles. Each archive contains the standard Esri shapefile component set for one
plot: `.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`, `.sbn`, `.sbx`, and `.shp.xml`
(FGDC metadata).

| Filename | Plot | Size |
|---|---|---|
| `B7A_plot_outline.tar.gz` | B7A | 97K |
| `C10_plot_outline.tar.gz` | C10 | 213K |
| `C6B_plot_outline.tar.gz` | C6B | 246K |
| `C7B_plot_outline.tar.gz` | C7B | 181K |
| `G3_plot_outline.tar.gz` | G3 | 222K |
| `I3B_plot_outline.tar.gz` | I3B | 135K |

### `eva02_base_cv0_fold_<N>_checkpoint_best.pt` (3 files)

Best-epoch model weights for the top-performing model configuration reported
in the associated manuscript (EVA-02-B encoder), for cross-validation folds
23, 24, and 25 of the CV0 (leave-one-year-out) splitting scheme.

| Filename | Size |
|---|---|
| `eva02_base_cv0_fold_23_checkpoint_best.pt` | 991M |
| `eva02_base_cv0_fold_24_checkpoint_best.pt` | 991M |
| `eva02_base_cv0_fold_25_checkpoint_best.pt` | 991M |

### `full_dataset.csv` / `flight_covariates.csv`

- `full_dataset.csv` (1.5M) — image-level metadata linking each plot image to
  its disease severity rating and acquisition details.
- `flight_covariates.csv` (4.8K) — flight-level environmental and acquisition
  covariates (e.g. date, plot, flight timing).

Column-level definitions:

**`full_dataset.csv`** — 26,071 rows × 8 columns. One row per scored plot image.

| Column | Type | Description |
|--------|------|-------------|
| `image_filename` | str | Image filename (`YYYYMMDD_FIELD_{PLOT:05d}.jpg`); primary join key |
| `plot` | int | Plot number |
| `field` | str | Field site (`G3`, `I3B`, `B7A`, `C6B`, `C10`, `C7B`) |
| `year` | int | Field season (2023, 2024, or 2025) |
| `date` | str | Visual scoring date (`M/D/YY`, unpadded, e.g. `7/12/23`) |
| `score` | float | SLB severity score (1–9 scale; half-integer increments) |
| `flight_date` | str | UAV flight date matched to scoring event (`M/D/YY`, unpadded) |
| `signed_days_diff` | int | Signed days between scoring and flight (−3 to +3; negative = flight before scoring) |

No missing-value codes are used in this file; every row is fully populated.

**`flight_covariates.csv`** — 27 rows × 16 columns. One row per flight matching
a raw imagery archive in this package.

| Column | Type | Description |
|--------|------|-------------|
| `flight_key` | str | Join key (`YYYYMMDD_FIELD`) |
| `flight_date` | str | Flight date (YYYY-MM-DD) |
| `field` | str | Field site identifier |
| `year` | int | Season (2023–2025) |
| `flight_start` | str | Flight start time (datetime string) |
| `flight_end` | str | Flight end time (datetime string) |
| `sza` | float | Solar zenith angle at flight midpoint (°) |
| `sun_azimuth` | float | Solar azimuth at flight midpoint (°) |
| `abs_time_from_noon` | float | Absolute minutes from solar noon |
| `irradiance_mean` | float | Mean shortwave irradiance during flight window (W m⁻²) |
| `irradiance_cv` | float | Coefficient of variation of irradiance (dimensionless) |
| `irradiance_n` | int | Number of minute-resolution irradiance observations in the flight window |
| `dpi` | int | Days post-inoculation at flight date |
| `agdd` | float | Accumulated growing degree days from planting (base 50 °F) |
| `row_azimuth` | float | Field row orientation (°) |
| `shadow_row_angle` | float | Angle between sun azimuth and row azimuth (°); proxy for shadow geometry |

No missing-value codes are used in this file; every row is fully populated.

## File formats and required software

- `.tar.gz` — standard gzipped tar archives; extract with `tar`, 7-Zip, or
  equivalent.
- `.jpg` — standard JPEG images; viewable with any image viewer.
- `.shp`/`.shx`/`.dbf`/`.prj`/`.cpg`/`.sbn`/`.sbx`/`.shp.xml` — Esri shapefile
  component set; open with GIS software (e.g. QGIS, ArcGIS, or the Python
  `geopandas`/`fiona` libraries).
- `.nav`/`.obs`/`.bin`/`.MRK` — raw GNSS observation, navigation, and
  timestamp-mark logs in receiver-native formats; process with PPK software
  (e.g. Emlid Studio, RTKLIB) or the DJI Terra workflow used for the source
  flights.
- `.pt` — PyTorch model checkpoint (`state_dict`); load with `torch.load()`
  (requires PyTorch and the EVA-02-B model definition from the associated
  [code repository](https://github.com/GageLab/uav4slb)).
- `.csv` — plain-text, comma-delimited, UTF-8; openable in any spreadsheet
  application or with `pandas.read_csv()`.

## Data collection and processing notes

- Raw flight folders exclude filesystem artifacts (`.DS_Store`,
  `_COPY_COMPLETE`, `_VERIFICATION_COMPLETE`) present in the original staging
  directory; these are transfer/processing markers with no scientific content.
- A small number of source flight folders that were confirmed to be exact
  duplicates (hardlinked copies) of other included flights were excluded to
  avoid redundant storage; no unique imagery or GNSS data was omitted.
- Raw flight archives range from 14–55 GB, above Dryad's general 10 GB/archive
  guideline. Files were not split further because each archive holds the
  complete, self-contained imagery and GNSS record for a single UAV flight;
  splitting would separate images from the GNSS logs needed to interpret
  them. Additionally, processing images uses all images captured from a
  given flight. This organization facilitates streamlined data structure
  for processing and metadata. 

## Related works

- Manuscript: Hammett, C. H., Rumley, K. R., Balint-Kurti, P. J., & Gage, J. L.
  (2026). Aerial imagery and deep learning accurately estimate maize foliar
  disease severity. *The Plant Phenome Journal*. Preprint on bioRxiv, 
  DOI: https://doi.org/10.64898/2026.06.03.729887

- Code repository: https://github.com/GageLab/uav4slb

## Funding

This work was supported by the North Carolina Corn Growers Association, USDA
NIFA Hatch project 7002327, and the North Carolina State University
Department of Entomology and Plant Pathology.

## Contact

Cole Hammett — Department of Entomology and Plant Pathology,
NC State University
PIs: Dr. Joseph L. Gage (jlgage@ncsu.edu), Department of Crop & Soil Sciences,
NC State University; Dr. Peter Balint-Kurti (pjbalint@ncsu.edu), Department of 
Entomology and Plant Pathology, NC State University & USDA-ARS

## License

Data are released under CC0 1.0 (Public Domain Dedication).
