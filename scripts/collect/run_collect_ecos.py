"""
한국은행 ECOS API 거시지표 수집 스크립트
수집 데이터: 기준금리, 주담대금리, M2, CPI, 부동산BSI
저장 위치: data/raw/ecos/
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

import requests
import pandas as pd
from dotenv import load_dotenv

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
load_dotenv()

API_KEY   = os.getenv("ECOS_API_KEY")
BASE_URL  = "https://ecos.bok.or.kr/api/StatisticSearch"
START     = "201301"
END       = "202512"
os.makedirs("data/raw/ecos", exist_ok=True)

# ── 수집할 통계 목록 ────────────────────────────────────────────
TARGETS = [
    {
        "name":        "기준금리",
        "stat_code":   "722Y001",   # 한국은행 기준금리 및 여수신금리
        "cycle":       "M",         # 월별
        "item_code1":  "0101000",   # 한국은행 기준금리
        "item_code2":  "",
        "value_col":   "base_rate",
        "unit":        "연%",
    },
    {
        "name":        "주담대금리",
        "stat_code":   "721Y001",   # 예금은행 대출금리(신규취급액)
        "cycle":       "M",
        "item_code1":  "BECBLA0302", # 주택담보대출
        "item_code2":  "",
        "value_col":   "mortgage_rate",
        "unit":        "연%",
    },
    {
        "name":        "M2통화량",
        "stat_code":   "101Y004",   # M2 상품별 구성내역(평잔, 계절조정)
        "cycle":       "M",
        "item_code1":  "BBHS00",    # M2 합계
        "item_code2":  "",
        "value_col":   "m2_bil_krw",
        "unit":        "십억원",
    },
    {
        "name":        "CPI",
        "stat_code":   "901Y009",   # 소비자물가지수(2020=100)
        "cycle":       "M",
        "item_code1":  "0",         # 총지수
        "item_code2":  "",
        "value_col":   "cpi",
        "unit":        "지수",
    },
    {
        "name":        "부동산BSI",
        "stat_code":   "512Y014",   # 업종별 기업경기실사지수(전망)
        "cycle":       "M",
        "item_code1":  "BA",        # 업황전망BSI
        "item_code2":  "L6800",     # 부동산업
        "value_col":   "bsi_realestate",
        "unit":        "지수",
    },
]


def fetch_ecos(target: dict) -> pd.DataFrame:
    """ECOS API 호출 → DataFrame 반환"""
    t = target
    # URL 구성: /StatisticSearch/{키}/json/{통계코드}/{주기}/{시작}/{종료}/{항목1}[/{항목2}]
    parts = [BASE_URL, API_KEY, "json", t["stat_code"],
             t["cycle"], START, END, t["item_code1"]]
    if t["item_code2"]:
        parts.append(t["item_code2"])
    url = "/".join(parts)

    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()

    # ECOS 응답 구조: {"StatisticSearch": {"list_total_count": N, "row": [...]}}
    ss = data.get("StatisticSearch", {})
    if "row" not in ss:
        raise ValueError(f"ECOS 응답 에러: {data}")

    rows = ss["row"]
    df = pd.DataFrame(rows)

    # 필요 컬럼: TIME (YYYYMM), DATA_VALUE
    df = df[["TIME", "DATA_VALUE"]].copy()
    df.columns = ["ym", t["value_col"]]
    df[t["value_col"]] = pd.to_numeric(df[t["value_col"]], errors="coerce")
    df = df[df[t["value_col"]].notna()].reset_index(drop=True)
    df["ym"] = df["ym"].astype(str).str.strip()

    return df


# ── 실행 ────────────────────────────────────────────────────────
print("=" * 60)
print("  ECOS 거시지표 수집 (2013.01 ~ 2025.12)")
print(f"  API KEY: {API_KEY[:6]}{'*' * 14}")
print("=" * 60)

results = {}
for t in TARGETS:
    print(f"\n[{t['name']}] 통계코드={t['stat_code']} 항목={t['item_code1']}")
    try:
        df = fetch_ecos(t)
        results[t["value_col"]] = df
        print(f"  ✅ {len(df)}건 수집  ({df['ym'].min()} ~ {df['ym'].max()})")
        print(f"     최신값: {df.iloc[-1]['ym']} = {df.iloc[-1][t['value_col']]} ({t['unit']})")

        # 개별 저장
        out = f"data/raw/ecos/{t['value_col']}_2013_2025.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"     저장: {out}")
    except Exception as e:
        print(f"  ❌ 실패: {e}")

# ── 전체 조인 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  전체 조인 → macro_ecos_2013_2025.csv")
print("=" * 60)

if results:
    # ym 기준으로 outer join
    dfs = list(results.values())
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="ym", how="outer")
    merged = merged.sort_values("ym").reset_index(drop=True)

    out_path = "data/raw/ecos/macro_ecos_2013_2025.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n  ✅ 조인 완료: {merged.shape[0]}행 × {merged.shape[1]}컬럼")
    print(f"  저장: {out_path}")
    print(f"\n  컬럼: {list(merged.columns)}")
    print(f"\n  최신 3개월:\n{merged.tail(3).to_string(index=False)}")
    print("\n  결측치 현황:")
    print(merged.isnull().sum().to_string())

print("\n" + "=" * 60)
print("  수집 완료")
print("=" * 60)
