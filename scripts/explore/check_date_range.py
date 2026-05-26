import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

REGIONS = ["seoul", "gyeonggi", "incheon"]

print(f"{'연도':<6} {'지역':<8} {'시작일':<12} {'종료일':<12} {'건수':>8}  {'비고'}")
print("─" * 60)

for year in range(2013, 2026):
    for region in REGIONS:
        path = f"data/raw/molit/apt_trade_{year}_{region}.csv"
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path, skiprows=15, encoding="cp949", dtype=str,
                         usecols=["계약년월", "계약일"])
        df.columns = df.columns.str.strip()
        df["deal_date"] = pd.to_datetime(
            df["계약년월"].str.strip() + df["계약일"].str.strip().str.zfill(2),
            format="%Y%m%d", errors="coerce"
        )
        df = df[df["deal_date"].notna()]

        min_date = df["deal_date"].min().strftime("%Y-%m-%d")
        max_date = df["deal_date"].max().strftime("%Y-%m-%d")
        count    = len(df)

        expected_start = f"{year}-01-01"
        expected_end   = f"{year}-12-31"

        note = ""
        if min_date > expected_start:
            note += f"⚠️ 시작 늦음({min_date})"
        if max_date < expected_end:
            note += f"⚠️ 종료 이름({max_date})"
        if not note:
            note = "✅"

        print(f"{year:<6} {region:<10} {min_date:<12} {max_date:<12} {count:>8,}  {note}")
