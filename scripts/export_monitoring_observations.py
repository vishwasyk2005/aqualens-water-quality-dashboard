"""Package quality-screened ESA cells for the bundled Vembanad monitoring area."""
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("esa_csv", type=Path, help="Dated ESA observations from the Vembanad extraction.")
    args = parser.parse_args()
    frame = pd.read_csv(args.esa_csv)
    coordinates = pd.read_csv(ROOT / "data/modis_observations.csv", usecols=[".geo"])
    positions = coordinates[".geo"].str.extract(r'"coordinates"\s*:\s*\[\s*([^,]+),\s*([^\]]+)\]')
    longitude, latitude = positions.astype(float).T.to_numpy()
    north_band = ((frame.esa_lat - latitude.min()) * 3 / (latitude.max() - latitude.min())).clip(0, 2).astype(int)
    east_band = ((frame.esa_lon - longitude.min()) * 2 / (longitude.max() - longitude.min())).clip(0, 1).astype(int)
    frame["zone_id"] = "Z" + (north_band * 2 + east_band + 1).astype(str)
    frame["cell_id"] = "ESA:" + frame.esa_lat.round(6).astype(str) + ":" + frame.esa_lon.round(6).astype(str)
    if frame.duplicated(["date", "cell_id"]).any():
        raise ValueError("ESA input must contain one reading per cell and date.")
    columns = ["date", "zone_id", "cell_id", "lake_surface_temp_c", "tsm_mg_l"]
    path = ROOT / "data" / "esa_vembanad_monitoring.csv.gz"
    frame[columns].to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"Packaged {frame.cell_id.nunique()} ESA cells for {frame.date.nunique()} archive dates.")


if __name__ == "__main__":
    main()
