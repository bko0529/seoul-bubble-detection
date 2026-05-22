## 국토부 실거래가 다운로드 방법

사이트: https://rt.molit.go.kr → 자료제공 → 아파트

### 매매 (apt_trade_YYYY.csv)
- 아파트 → 매매 탭 선택
- 계약일자: YYYY-01-01 ~ YYYY-12-31
- 지번주소 / 시도: 서울특별시 / 시군구: 전체
- CSV 다운로드
- 파일명: `apt_trade_2013.csv` ~ `apt_trade_2025.csv`

### 전월세 (apt_jeonse_YYYY.csv)
- 아파트 → 전월세 탭 선택
- 동일 조건으로 다운로드
- 파일명: `apt_jeonse_2013.csv` ~ `apt_jeonse_2025.csv`

### 주의사항
- `skiprows=15` 로 읽어야 실제 데이터 시작
- 거래금액이 문자열+쉼표 형태 → 전처리 필요
- 해제사유발생일 != '-' 인 행 = 계약취소 → 제거
