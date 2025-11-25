import pandas as pd
import numpy as np
from pathlib import Path

# Path
DATA_PATH = "AirQualityUCI.csv"
# 5 pollutants
TARGET_COLS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

def load_and_preprocess_air_quality(csv_path: Path) -> pd.DataFrame:
    """
    Read and preprocess raw AirQualityUCI.csv:
      - Handle ; separator and comma decimals
      - Merge Date+Time into Datetime index
      - Treat -200 as missing and interpolate
      - Keep only variables used in the project
    Returns: DataFrame sorted by time with no obvious missing values, index is Datetime.
    """
    # Raw data uses ; as separator, decimals use ,
    df_raw = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        header=0
    )

    # Remove columns that may be empty at the end (many versions have unnamed columns)
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains("^Unnamed")]

    # Merge Date + Time -> Datetime
    # Original Date format is DD/MM/YYYY
    df_raw['Datetime'] = pd.to_datetime(
        df_raw['Date'] + ' ' + df_raw['Time'],
        format='%d/%m/%Y %H.%M.%S',
        errors='coerce'
    )
    df_raw = df_raw.dropna(subset=['Datetime'])
    df_raw = df_raw.set_index('Datetime').sort_index()

    # Keep only variables we care about (5 pollutants + meteorological variables)
    cols_keep = TARGET_COLS + ["T", "RH", "AH"]
    df = df_raw[cols_keep].copy()

    # -200 treated as missing
    cols_numeric = df.columns.tolist()
    for c in cols_numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df.loc[df[c] == -200, c] = np.nan

    # Interpolate by time first, then forward and backward fill
    df = df.sort_index()
    df = df.interpolate(method="time")
    df = df.ffill().bfill()

    # Remove rows that are still completely empty (safety check)
    df = df.dropna(how="any")

    return df

df = load_and_preprocess_air_quality(DATA_PATH)
print("Data shape after preprocessing:", df.shape)
print(df.head())

df.to_csv("clean_air_quality.csv", index=True)