import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

files = [
    ("경기", "data/raw/molit/apt_trade_2025_gyeonggi.csv"),
    ("인천", "data/raw/molit/apt_trade_2025_incheon.csv"),
]

for region, fpath in files:
    df = pd.read_csv(fpath, skiprows=15, encoding="cp949", dtype=str, nrows=3)
    df.columns = df.columns.str.strip()
    print(f"[{region}] 컬럼 목록:")
    print("  " + ", ".join(df.columns.tolist()))
    print(f"[{region}] 시군구 샘플:")
    for v in df["시군구"].tolist():
        print(f"  {v}")
    print()
