import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

path = "data/raw/molit/apt_jeonse_2025_seoul.csv"
df = pd.read_csv(path, skiprows=15, encoding="cp949", dtype=str, nrows=5)
df.columns = df.columns.str.strip()

print("[컬럼 목록]")
for c in df.columns:
    print(f"  {c}")

print()
print("[샘플 데이터 - 3행]")
print(df.head(3).to_string())
