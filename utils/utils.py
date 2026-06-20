"""
Download and process Himawari-8/9 Target-area satellite data for IR→VIS translation.

Himawari-8:  2020/01/01 00:00 → 2022/11/30 23:30  (sat at 140.7°E)
Himawari-9:  2022/12/01 00:00 → 2025/12/31 23:30  (sat at 140.7°E)

Downloads every 30 min.  One target-area segment (R301) per band per timestamp.
  B03 (VIS, R05 / 500 m)  → target
  B08, B09, B10, B11, B13, B15, B16 (IR, R20 / 2 km) → input

Each band image is rescaled to 512×512 and saved as .npy + visualisation .jpg.
Solar zenith/azimuth and satellite zenith/azimuth are computed on the VIS native
grid, resized to 512×512 raw degrees, and saved.  Normalisation is left to the
DataLoader.  Raw .DAT files are deleted after processing.

Usage:
  python utils.py                     # full batch (years of 60-min data ~26k pairs)
  python utils.py --test              # smoke-test on a single timestamp
  python utils.py --date 202604120320 # process one specific UTC timestamp
"""

import os
import argparse
import numpy as np
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import bz2
import shutil

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import satpy

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAT_LON = 140.7                     # °E  — both Himawari-8 and -9
SAT_HEIGHT = 35786000.0             # geostationary altitude above equator (m)
GEO_RADIUS = 6378137.0 + SAT_HEIGHT # Earth-centre → satellite distance (m)

HIMAWARI8_START = datetime(2020, 1, 1, 0, 0)
HIMAWARI8_END   = datetime(2022, 11, 30, 23, 30)
HIMAWARI9_START = datetime(2022, 12, 1, 0, 0)
HIMAWARI9_END   = datetime(2025, 12, 31, 23, 30)

VIS_BAND = 3
#IR_BANDS = [8, 9, 10, 11, 13, 15, 16]
IR_BANDS = [11, 13, 15]

# Band → resolution code  (R05 = 500 m,  R20 = 2 km)
#BAND_RES = {3: 5, 8: 20, 9: 20, 10: 20, 11: 20, 13: 20, 15: 20, 16: 20}
BAND_RES = {3: 5, 11: 20, 13: 20, 15: 20}

TARGET_SIZE = 512
S3_PREFIX = "AHI-L1b-Target"
SEGMENT = 1                         # R301 only (one target-area scan per timestamp)

# Himawari-8 and -9 data live in separate NOAA S3 buckets
BUCKETS = {"H8": "noaa-himawari8", "H9": "noaa-himawari9"}

# WGS84 ellipsoid
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F


# ===================================================================
# S3 download
# ===================================================================
def download_one_file(date_time, satellite, band, local_dir="data"):
    """
    Download a single R301 .DAT.bz2 file from NOAA S3, decompress, return .DAT path.

    Parameters
    ----------
    date_time : datetime (naive UTC)
    satellite : "H8" | "H9"
    band : int  (3, 8, 9, …)
    local_dir : str

    Returns
    -------
    str or None  — path to decompressed .DAT file, or None if unavailable.
    """
    os.makedirs(local_dir, exist_ok=True)

    sat_prefix = "H08" if satellite == "H8" else "H09"
    band_str = f"B{band:02d}"
    res_str = f"R{BAND_RES[band]:02d}"
    ts_file = date_time.strftime("%Y%m%d_%H%M")
    prefix = date_time.strftime(f"{S3_PREFIX}/%Y/%m/%d/%H%M/")

    fname = f"HS_{sat_prefix}_{ts_file}_{band_str}_R30{SEGMENT}_{res_str}_S0101.DAT.bz2"
    s3_key = prefix + fname
    local_bz2 = os.path.join(local_dir, fname)
    local_dat = local_bz2[:-4]

    if os.path.exists(local_dat):
        return local_dat

    bucket = BUCKETS[satellite]
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    try:
        client.download_file(bucket, s3_key, local_bz2)
        with bz2.BZ2File(local_bz2, "rb") as f_in:
            with open(local_dat, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(local_bz2)
        # print("Download complete")
        return local_dat
    except Exception as e:
        if os.path.exists(local_bz2):
            os.remove(local_bz2)
        return None


# ===================================================================
# Solar position  (NOAA equations — identical to src/utils.py)
# ===================================================================
def _julian_day(dt):
    year, month, day = dt.year, dt.month, dt.day
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    frac = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    return jd + frac


def solar_position_noaa(utc_time, lat, lon):
    """Solar zenith (°) and azimuth (°) via NOAA solar calculator."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    jd = _julian_day(utc_time)
    jc = (jd - 2451545.0) / 36525.0

    geom_mean_long = (280.46646 + 36000.76983 * jc + 0.0003032 * jc * jc) % 360
    geom_mean_anom = (357.52911 + 35999.05029 * jc - 0.0001537 * jc * jc) % 360
    e = 0.016708634 - 0.000042037 * jc - 0.0000001267 * jc * jc

    sin1 = np.sin(np.radians(geom_mean_anom))
    sin2 = np.sin(np.radians(2 * geom_mean_anom))
    sin3 = np.sin(np.radians(3 * geom_mean_anom))
    eq_center = (
        sin1 * (1.914602 - 0.004817 * jc - 0.000014 * jc * jc)
        + sin2 * (0.019993 - 0.000101 * jc)
        + sin3 * 0.000289
    )

    sun_true_long = geom_mean_long + eq_center
    omega = 125.04 - 1934.136 * jc
    sun_app_long = sun_true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))

    obliquity = (
        23 + 26 / 60.0 + 21.448 / 3600.0
        - (46.815 * jc + 0.00059 * jc * jc - 0.001813 * jc * jc * jc) / 3600.0
    )
    obliquity_rad = np.radians(obliquity)

    dec = np.degrees(np.arcsin(np.sin(np.radians(sun_app_long)) * np.sin(obliquity_rad)))
    dec_rad = np.radians(dec)

    y = np.tan(obliquity_rad / 2.0) ** 2
    eot = 4.0 * np.degrees(
        y * np.sin(2 * np.radians(geom_mean_long))
        - 2 * e * np.sin(np.radians(geom_mean_anom))
        + 4 * e * y * np.sin(np.radians(geom_mean_anom)) * np.cos(2 * np.radians(geom_mean_long))
        - 0.5 * y * y * np.sin(4 * np.radians(geom_mean_long))
        - 1.25 * e * e * np.sin(2 * np.radians(geom_mean_anom))
    )

    utc_hour = utc_time.hour + utc_time.minute / 60.0 + utc_time.second / 3600.0
    solar_time_min = (utc_hour * 60.0 + eot + 4.0 * lon) % 1440.0
    hour_angle = (solar_time_min / 4.0) - 180.0
    ha_rad = np.radians(hour_angle)

    lat_rad = np.radians(lat)
    cos_zenith = (
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith = np.degrees(np.arccos(cos_zenith))

    S_east = -np.cos(dec_rad) * np.sin(ha_rad)
    S_north = np.sin(dec_rad) * np.cos(lat_rad) - np.cos(dec_rad) * np.cos(ha_rad) * np.sin(lat_rad)
    azimuth = np.degrees(np.arctan2(S_east, S_north)) % 360.0
    
    # print(f"solar zenith range: {zenith.min()}-{zenith.max()}")
    # print(f"solar azimuth range: {azimuth.min()}-{azimuth.max()}")
    return zenith.astype(np.float32), azimuth.astype(np.float32)


# ===================================================================
# Satellite viewing geometry
# ===================================================================
def compute_satellite_angles(lat, lon):
    """
    Satellite zenith (°) and azimuth (°) for each pixel.

    Satellite is at SAT_LON °E, 0°N, GEO_RADIUS from Earth centre (WGS-84).
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    sat_lon_rad = np.radians(SAT_LON)

    # pixel ECEF (h = 0)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat_rad) ** 2)
    X_px = N * np.cos(lat_rad) * np.cos(lon_rad)
    Y_px = N * np.cos(lat_rad) * np.sin(lon_rad)
    Z_px = N * (1.0 - WGS84_E2) * np.sin(lat_rad)

    # satellite ECEF
    X_sat = GEO_RADIUS * np.cos(sat_lon_rad)
    Y_sat = GEO_RADIUS * np.sin(sat_lon_rad)
    Z_sat = 0.0

    # vector pixel → satellite
    dX = X_sat - X_px
    dY = Y_sat - Y_px
    dZ = Z_sat - Z_px

    # rotate to local ENU
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    E = -dX * sin_lon + dY * cos_lon
    N_enu = -dX * sin_lat * cos_lon - dY * sin_lat * sin_lon + dZ * cos_lat
    U = dX * cos_lat * cos_lon + dY * cos_lat * sin_lon + dZ * sin_lat

    dist = np.sqrt(dX ** 2 + dY ** 2 + dZ ** 2)
    sat_zenith = np.degrees(np.arccos(np.clip(U / dist, -1.0, 1.0)))
    sat_azimuth = np.degrees(np.arctan2(E, N_enu)) % 360.0
    # print(f"satellite zenith range: {sat_zenith.min()}-{sat_zenith.max()}")
    # print(f"satellite azimuth range: {sat_azimuth.min()}-{sat_azimuth.max()}")
    return sat_zenith.astype(np.float32), sat_azimuth.astype(np.float32)


# ===================================================================
# Image helpers
# ===================================================================
def _to_uint8(data, vmin=None, vmax=None):
    """Min-max normalise 2-D array → [0, 255] uint8, with NaN guard."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    v_min = vmin if vmin else finite.min()
    v_max = vmax if vmax else finite.max()
    if v_max == v_min:
        return np.zeros_like(data, dtype=np.uint8)
    # data = np.clip(data, v_min, v_max)
    norm = (data - v_min) / (v_max - v_min) * 255.0
    norm = np.nan_to_num(norm, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(norm, 0, 255).astype(np.uint8)


def rescale_to_512(data, type: str):
    """Resize a 2-D array → (512, 512) uint8 via PIL LANCZOS."""
    if type == 'ir':
        img = Image.fromarray(_to_uint8(data, vmin=173.15, vmax=323.15)) # IR-BW scale brightness temperature [173.15, 323.15]
    elif type == 'vis':
        # print(f"VIS data range: min:{data.min():06f}, max:{data.max():06f}")
        img = Image.fromarray(_to_uint8(data, vmin=0, vmax=125)) # reflectance [0, 100+]
    else:
        raise ValueError(f"Unknown data category: {type}")
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)

'''
def normalise_to_01(data):
    """Min-max normalise 2-D array → [0, 1] float32."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    vmin, vmax = finite.min(), finite.max()
    if vmax == vmin:
        return np.zeros_like(data, dtype=np.float32)
    norm = (data - vmin) / (vmax - vmin)
    return np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
'''

# ===================================================================
# Core processing
# ===================================================================
def load_band_data(dat_path, band):
    """
    Load a single .DAT file with satpy, return the xarray DataArray.

    Returns None on failure.
    """
    try:
        scene = satpy.Scene(reader="ahi_hsd", filenames=[dat_path])
        scene.load([f"B{band:02d}"])
        return scene[f"B{band:02d}"]
    except Exception as e:
        print(f"      satpy error on {os.path.basename(dat_path)}: {e}")
        return None


def compute_angles_from_data(data_array, day_only=True):
    """
    Compute solar + satellite zenith/azimuth on *data_array*'s native grid,
    resize to 512×512 float32 (raw degrees, no normalisation).

    Returns float32 (4, 512, 512).
    """
    area = data_array.attrs["area"]
    lon, lat = area.get_lonlats()

    tp = data_array.attrs["time_parameters"]
    mid_time = tp["observation_start_time"] + (
        tp["observation_end_time"] - tp["observation_start_time"]
    ) / 2

    sol_zen, sol_az = solar_position_noaa(mid_time, lat, lon)
    if day_only and sol_zen.max() > 90.0:
        return None
    sat_zen, sat_az = compute_satellite_angles(lat, lon)

    angle_maps = []
    for arr in [sol_zen, sol_az, sat_zen, sat_az]:
        pil_img = Image.fromarray(arr.astype(np.float32))
        pil_img = pil_img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        angle_maps.append(np.array(pil_img, dtype=np.float32))

    return np.stack(angle_maps, axis=0)  # (4, 512, 512)


def process_timestamp(date_time, satellite, output_dir="data", save_jpg=False, save_angles=True):
    """
    Download + process all bands for one UTC timestamp.

    Saves
    -----
    data/vis/    {sat}_vis_{yymmdd-HHMM}.npy       uint8  (512, 512)
    data/ir/     {sat}_ir_{yymmdd-HHMM}.npy        uint8  (7, 512, 512)
    data/angles/ {sat}_angles_{yymmdd-HHMM}.npy    float32 (4, 512, 512) — raw degrees
    data/jpg/    himawari-{sat}-b{xx}-{yymmdd-HHMM}.jpg

    Returns True on success.
    """
    ts = date_time.strftime("%Y%m%d_%H%M")
    sat_lower = "h8" if satellite == "H8" else "h9"
    ts_fmt = date_time.strftime("%y%m%d-%H%M")
    print(f"  [{satellite}] {ts}")

    # JPG directory (used for B03, IR bands, and angle visualisations)
    jpg_dir = os.path.join(output_dir, "jpg")
    os.makedirs(jpg_dir, exist_ok=True)

    # ---- IR bands (input channels) ----
    ir_channels = []
    ir_files = []
    for i, band in enumerate(IR_BANDS):
        f = download_one_file(date_time, satellite, band, output_dir)
        if f is None:
            print(f"    B{band:02d}: download failed — skipping timestamp")
            return False
        ir_files.append(f)

        data = load_band_data(f, band)
        if data is None:
            _cleanup(ir_files)
            return False

        if i == 0:
            angles_512 = compute_angles_from_data(data)
            if angles_512 is None: # not during daytime
                _cleanup(ir_files)
                return False
            
            # save angles
            if save_angles:
                ang_dir = os.path.join(output_dir, "angles")
                os.makedirs(ang_dir, exist_ok=True)
                np.save(os.path.join(ang_dir, f"{sat_lower}_angles_{ts_fmt}.npy"), angles_512)

                # save angle JPGs (zenith: 0°→-1, 90°→1  |  azimuth: -180°→-1, 180°→1)
                if save_jpg:
                    for i, name in enumerate(["sol_zen", "sol_az", "sat_zen", "sat_az"]):
                        arr = angles_512[i]
                        if name == "sol_az" or name == "sat_az":
                            arr = np.where(arr > 180, arr - 360, arr)  # [0,360] → [-180,180]
                            norm = arr / 180.0                           # [-1, 1]
                        else:
                            norm = 2.0 * arr / 90.0 - 1.0               # [0,90] → [-1, 1]
                        # normalize = plt.Normalize(vmin=-1, vmax=1)
                        # disp = normalize(norm)
                        # disp = ((norm + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
                        plt.imsave(
                            os.path.join(jpg_dir, f"himawari-{sat_lower}-{name}-{ts_fmt}.jpg"),
                            norm, cmap="inferno", vmin=-1, vmax=1
                        )
        
        ir_512 = rescale_to_512(data.values, 'ir')
        ir_channels.append(ir_512)

        # JPG
        if save_jpg:
            plt.imsave(
                os.path.join(jpg_dir, f"himawari-{sat_lower}-b{band:02d}-{ts_fmt}.jpg"),
                ir_512, cmap="gray_r",
            )

    ir_stack = np.stack(ir_channels, axis=0).astype(np.uint8)  # (3, 512, 512)
    ir_dir = os.path.join(output_dir, "ir")
    os.makedirs(ir_dir, exist_ok=True)
    np.save(os.path.join(ir_dir, f"{sat_lower}_ir_{ts_fmt}.npy"), ir_stack)

    # ---- B03 (VIS target) ----
    vis_file = download_one_file(date_time, satellite, VIS_BAND, output_dir)
    if vis_file is None:
        print(f"    B03: download failed — skipping timestamp")
        return False

    vis_data = load_band_data(vis_file, VIS_BAND)
    if vis_data is None:
        _cleanup([vis_file])
        return False

    vis_512 = rescale_to_512(vis_data.values, 'vis')

    # save VIS
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    np.save(os.path.join(vis_dir, f"{sat_lower}_vis_{ts_fmt}.npy"), vis_512)

    if save_jpg:
        # save B03 JPG
        plt.imsave(
            os.path.join(jpg_dir, f"himawari-{sat_lower}-b03-{ts_fmt}.jpg"),
            vis_512, cmap="gray",
        )

    # ---- cleanup ----
    _cleanup([vis_file] + ir_files)
    print(f"    ✓ saved {sat_lower} {ts_fmt}")
    return True


def _cleanup(file_list):
    for f in file_list:
        try:
            if f and os.path.exists(f):
                os.remove(f)
        except OSError:
            pass


# ===================================================================
# Time generation
# ===================================================================
def _satellite_for(dt):
    if HIMAWARI8_START <= dt <= HIMAWARI8_END:
        return "H8"
    if HIMAWARI9_START <= dt <= HIMAWARI9_END:
        return "H9"
    return None


def generate_timestamps(start_time=None, end_time=None):
    """Yield (datetime, satellite) every 30 min across the full H8+H9 range."""
    t = datetime.strptime(start_time, "%Y%m%d%H%M") if start_time else HIMAWARI8_START
    end = datetime.strptime(end_time, "%Y%m%d%H%M") if end_time else HIMAWARI9_END
    while t <= HIMAWARI8_END and t <= end:
        yield t, "H8"
        t += timedelta(minutes=60)
    t = HIMAWARI9_START
    while t <= end:
        yield t, "H9"
        t += timedelta(minutes=60)


# ===================================================================
# Multithreaded year-by-year download
# ===================================================================
_progress_lock = threading.Lock()
_global_success = 0


def _year_timestamps(year):
    """
    Yield (datetime, satellite) for every 60-min slot in *year* across
    the valid Himawari-8 / Himawari-9 windows.
    """
    y_start = datetime(year, 1, 1, 0, 0)
    y_end = datetime(year, 12, 31, 23, 30)

    if year < 2020 or year > 2025:
        return

    # Clip to each satellite's active window
    if year <= 2022:
        t = max(y_start, HIMAWARI8_START)
        stop = min(y_end, HIMAWARI8_END)
        while t <= stop:
            yield t, "H8"
            t += timedelta(minutes=60)

    if year >= 2022:
        t = max(y_start, HIMAWARI9_START)
        stop = min(y_end, HIMAWARI9_END)
        while t <= stop:
            yield t, "H9"
            t += timedelta(minutes=60)


def _download_year(year, output_dir, save_angles=True):
    """
    Worker: download and process all timestamps for a single year.
    Prints progress with the year label.
    """
    global _global_success
    local_success = 0
    for dt, sat in _year_timestamps(year):
        try:
            if process_timestamp(dt, sat, output_dir, save_angles=save_angles):
                local_success += 1
        except KeyboardInterrupt:
            return local_success
        except Exception:
            pass

    with _progress_lock:
        _global_success += local_success
    print(f"  [year {year}] finished — {local_success} timestamps saved")
    return local_success


def main_threaded(years, output_dir="data", num_workers=None, save_angles=True):
    """
    Download multiple years in parallel using a thread pool.

    Parameters
    ----------
    years : list[int]
        Years to download (e.g., [2021, 2022, 2023, 2024, 2025]).
    output_dir : str
    num_workers : int or None
        Number of threads (default: len(years)).
    """
    years = sorted(set(years))
    num_workers = num_workers or len(years)
    print(f"Starting {num_workers}-thread download for years {years}")
    print(f"Output directory: {os.path.abspath(output_dir)}")

    global _global_success
    _global_success = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_download_year, y, output_dir, save_angles): y
            for y in years
        }
        try:
            for f in as_completed(futures):
                y = futures[f]
                try:
                    f.result()
                except Exception as e:
                    print(f"  [year {y}] thread crashed: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted — waiting for in-flight downloads to finish...")
            executor.shutdown(wait=True, cancel_futures=True)
            print(f"Partial results saved ({_global_success} timestamps).")
            return

    print(f"\nDone.  {_global_success} total timestamps processed.")


# ===================================================================
# Main entry-points
# ===================================================================
def main_full(output_dir="data", start_time=None, end_time=None):
    """Full pipeline — years of 60-min data.  Runs until interrupted or done."""
    print("Himawari-8/9 full pipeline — press Ctrl+C to stop")
    success = 0
    for dt, sat in generate_timestamps(start_time=start_time, end_time=end_time):
        try:
            if process_timestamp(dt, sat, output_dir):
                success += 1
                if success % 100 == 0:
                    print(f"{success} timestamps processed.")
        except KeyboardInterrupt:
            print(f"\nInterrupted.  {success} timestamps processed.")
            return
        except Exception as e:
            print(f"  !! error @ {dt} {sat}: {e}")
    print(f"\nDone.  {success} timestamps processed.")


def main_test(output_dir="data"):
    """Smoke-test on a single known-good timestamp."""
    print("Test mode — processing single timestamp …")
    # Try a recent H9 date; fall back to another if missing
    for test_dt in [
        datetime(2025, 12, 31, 0, 0),
        datetime(2025, 6, 15, 6, 0),
        datetime(2024, 1, 15, 3, 0),
    ]:
        sat = _satellite_for(test_dt)
        if sat is None:
            sat = "H9" if test_dt >= HIMAWARI9_START else "H8"
        print(f"Trying {test_dt} ({sat}) …")
        if process_timestamp(test_dt, sat, output_dir, save_jpg=True):
            return
        print("  (no data — trying next fallback)")
    print("All test dates failed.  Try --date with a known timestamp.")


def main_date(date_str, output_dir="data"):
    """Process a single UTC timestamp: yyyymmddHHMM  (e.g. 202604120320)."""
    dt = datetime.strptime(date_str, "%Y%m%d%H%M")
    sat = _satellite_for(dt)
    if sat is None:
        sat = "H9" if dt >= HIMAWARI9_START else "H8"
    process_timestamp(dt, sat, output_dir, save_jpg=True)


# ===================================================================
# CLI
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Himawari-8/9 target-area download + 512×512 preprocessing"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Smoke-test: process one timestamp and exit",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Process single timestamp: yyyymmddHHMM (e.g. 202604120320)",
    )
    parser.add_argument(
        "--out", type=str, default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--begin_time", type=str, default=None,
        help="Manually assign the starting time of the download"
    )
    parser.add_argument(
        "--end_time", type=str, default=None,
        help="Manually assign the ending time of the download"
    )
    parser.add_argument(
        "--years", type=str, default=None,
        help="Comma-separated years for multi-threaded download (e.g. '2021,2022,2023,2024,2025')"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel threads (default: number of years)"
    )
    args = parser.parse_args()

    if args.test:
        main_test(args.out)
    elif args.date:
        main_date(args.date, args.out)
    elif args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
        main_threaded(years, args.out, args.workers, save_angles=False)
    else:
        main_full(args.out, args.begin_time, args.end_time)
