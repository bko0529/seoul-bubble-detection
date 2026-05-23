# CHANGELOG

## [Unreleased]

### Added
- 전월세 전처리 완료 (2013~2025, 수도권 6,918,409건)
- 매매 × 전월세 피처 조인 (features_sudogwon.parquet, 45,996행 × 22컬럼)
- 버블 Pseudo-label 생성 (정상/과열/버블)
- scripts/ 폴더 구조 정리 (explore/, preprocess/)

### Fixed
- 경기도 주소 파싱 버그 수정 예정 (gu_idx=2 단순 적용 문제)

---

## [0.2.0] — 2025-05-22

### Added
- 수도권(서울·경기·인천) 확장
- 매매 전처리 완료 (2013~2025, 3,340,896건)
- 날짜 범위 검증 (39개 파일 전수 확인)
- 전월세 파일명 표준화 스크립트

### Changed
- 프로젝트명: seoul → sudogwon
- REGION_CONFIG 멀티지역 파싱 구조 도입

---

## [0.1.0] — 2025-05-22

### Added
- 프로젝트 초기 구조 설정
- PostgreSQL 스키마 설계
- 브랜치 전략 수립 (main / dev-kangwook / dev-sihwan)
