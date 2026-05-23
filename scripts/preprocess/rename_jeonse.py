import sys, os, re
sys.stdout.reconfigure(encoding="utf-8")

folder = "c:/Users/bko05/Desktop/seoul-bubble-detection/data/raw/molit"

for fname in os.listdir(folder):
    # 이미 정리된 파일은 스킵
    if re.match(r"apt_jeonse_\d{4}_(seoul|gyeonggi|incheon)\.csv", fname):
        continue

    # 연도 추출 (4자리 숫자)
    year_match = re.search(r"(20\d{2})", fname)
    if not year_match:
        continue

    year = year_match.group(1)

    # 지역 판별
    if "경기" in fname:
        region = "gyeonggi"
    elif "인천" in fname:
        region = "incheon"
    else:
        continue

    # 전월세 파일만 처리
    if "전월세" not in fname:
        continue

    new_name = f"apt_jeonse_{year}_{region}.csv"
    old_path = os.path.join(folder, fname)
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)
    print(f"{fname}  →  {new_name}")

print("\n완료!")
