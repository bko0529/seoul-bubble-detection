import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

os.chdir("c:/Users/bko05/Desktop/seoul-bubble-detection")

# 서울 2013, 서울 2020, 경기 2013 세 파일 비교
files = [
    ("2013 서울",    "data/raw/molit/apt_jeonse_2013_seoul.csv"),
    ("2020 서울",    "data/raw/molit/apt_jeonse_2020_seoul.csv"),
    ("2013 경기",    "data/raw/molit/apt_jeonse_2013_gyeonggi.csv"),
    ("2013 인천",    "data/raw/molit/apt_jeonse_2013_incheon.csv"),
]

for label, path in files:
    df = pd.read_csv(path, skiprows=15, encoding="cp949", dtype=str, nrows=3)
    df.columns = df.columns.str.strip()
    print(f"\n{'='*60}")
    print(f"[{label}] 컬럼: {list(df.columns)}")
    print(df.head(2).to_string())
