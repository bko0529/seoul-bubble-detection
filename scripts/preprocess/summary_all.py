import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

summary_rows = []
total_all = 0

for year in range(2013, 2026):
    path = f"data/processed/apt_trade_{year}_sudogwon_clean.csv"
    if not os.path.exists(path):
        print(f"⚠️  {year} 파일 없음 → 스킵")
        continue

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["거래금액(만원)"]    = pd.to_numeric(df["거래금액(만원)"],    errors="coerce")
    df["㎡당단가(만원)"] = pd.to_numeric(df["㎡당단가(만원)"], errors="coerce")

    sido_counts = df["시도"].value_counts()
    total_all  += len(df)

    summary_rows.append({
        "연도":           year,
        "합계":           len(df),
        "서울":           sido_counts.get("서울", 0),
        "경기":           sido_counts.get("경기", 0),
        "인천":           sido_counts.get("인천", 0),
        "중위가격(만원)":  int(df["거래금액(만원)"].median()),
        "중위단가(만원/m2)": round(df["㎡당단가(만원)"].median(), 1),
    })

summary_df = pd.DataFrame(summary_rows)

print(f"{'='*65}")
print(f"  수도권 매매 전처리 전체 요약 (2013~2025)")
print(f"{'='*65}")
print(summary_df.to_string(index=False))
print(f"{'='*65}")
print(f"  총 합계: {total_all:,}건")
print(f"{'='*65}")
