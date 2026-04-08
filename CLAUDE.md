# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

교통사고 관련 판례를 검색하는 RAG(Retrieval-Augmented Generation) 시스템. 국가법령정보센터 API에서 판례를 수집하고, BGE-M3 임베딩 + Qdrant 벡터DB로 유사 판례를 검색한다.

## Commands

```bash
# 판례 수집 → 임베딩 → Qdrant 저장 (법령 API 호출, 시간 소요)
python3 rag_engine.py ingest

# 유사 판례 검색
python3 rag_engine.py query "검색 질문"

# 의존성 설치
pip install -r requirements.txt
```

## Architecture

단일 파일(`rag_engine.py`) 구조:

1. **데이터 수집** — 국가법령정보센터 REST API(`law.go.kr/DRF`)에서 교통사고 관련 키워드로 판례 검색 → 상세 본문 조회. API 키는 `.env`의 `OC` 변수.
2. **전처리** — HTML 태그 제거, 교통사고 관련 키워드 필터링, `사건명 + 판시사항 + 판결요지`를 임베딩용 텍스트로 구성.
3. **임베딩 & 저장** — `BAAI/bge-m3` 모델(dense 1024차원, fp16)로 임베딩하여 Qdrant 로컬 파일DB(`./qdrant_data`)에 저장. 컬렉션명: `traffic_precedents`.
4. **검색** — Dense 벡터 검색으로 상위 100건 후보를 가져온 뒤, BGE-M3의 `compute_score`(dense+sparse+colbert 가중 합산)로 리랭킹하여 최종 top_k건 반환.

## Key Details

- Qdrant는 로컬 파일 모드(`./qdrant_data`). 서버 불필요.
- BGE-M3 모델은 첫 실행 시 HuggingFace에서 자동 다운로드(~2.2GB).
- 법령 API는 rate limit이 있어 요청 간 0.3초 sleep 적용.
- `compute_score`의 `weights_for_different_modes`는 `[dense, sparse, colbert]` 순서.
