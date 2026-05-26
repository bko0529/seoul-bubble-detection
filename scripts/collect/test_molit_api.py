"""
국토교통부 통계누리 API 연결 테스트 & 사용 가능한 통계 확인
stat.molit.go.kr
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="c:/Users/bko05/Desktop/seoul-bubble-detection/.env")
API_KEY = os.getenv("MOLIT_STAT_API_KEY")

BASE_URL = "https://stat.molit.go.kr/statPortal/cate/openApiView.do"

# ── 테스트할 통계 항목 코드 ──────────────────────────────────────
# 아파트 공급 관련 주요 itemId 후보
ITEMS = {
    "주택건설인허가_전국":  "42",
    "주택준공_전국":         "44",
    "미분양_전국":           "61",
    "입주물량_예정":         "73",
}

def fetch_stat(item_id, item_name):
    params = {
        "key":    API_KEY,
        "itemId": item_id,
        "gubun":  "total",    # 합계
        "region": "11",       # 서울 (시도 코드)
        "format": "json",
        "numOfRows": 5,
        "pageNo": 1,
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        print(f"\n✅ [{item_name}] itemId={item_id}")
        print(f"   status: {r.status_code}")
        # 응답 구조 출력
        if isinstance(data, dict):
            print(f"   keys: {list(data.keys())}")
            # 첫 번째 레코드
            for k, v in data.items():
                if isinstance(v, list) and v:
                    print(f"   {k}[0]: {v[0]}")
                    break
                elif isinstance(v, dict):
                    print(f"   {k}: {v}")
        else:
            print(f"   응답: {str(data)[:200]}")
    except Exception as e:
        print(f"\n❌ [{item_name}] itemId={item_id} → {e}")
        print(f"   응답 텍스트: {r.text[:300] if 'r' in dir() else 'N/A'}")

print("=" * 60)
print("  국토교통부 통계누리 API 연결 테스트")
print(f"  API KEY: {API_KEY[:6]}{'*'*14}")
print("=" * 60)

for name, iid in ITEMS.items():
    fetch_stat(iid, name)

print("\n" + "=" * 60)
