from __future__ import annotations
import os
import re
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

# ── 설정 ──
OC = os.getenv("OC")
SEARCH_URL = "http://www.law.go.kr/DRF/lawSearch.do"
DETAIL_URL = "http://www.law.go.kr/DRF/lawService.do"
COLLECTION_NAME = "traffic_precedents"
EMBEDDING_DIM = 1024  # BGE-M3 default dimension
SCORE_THRESHOLD = 0.0  # 항상 가장 유사한 판례 반환

# 재시도 설정이 포함된 세션
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))


# ── 1. 판례 데이터 수집 ──

def search_precedents(query: str, display: int = 20, page: int = 1) -> list[dict]:
    """판례 목록 검색 API 호출"""
    params = {
        "OC": OC,
        "target": "prec",
        "type": "JSON",
        "query": query,
        "display": display,
        "page": page,
    }
    resp = session.get(SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    prec_list = data.get("PrecSearch", {}).get("prec", [])
    # 단건이면 dict로 올 수 있으므로 리스트로 통일
    if isinstance(prec_list, dict):
        prec_list = [prec_list]
    return prec_list


def get_precedent_detail(prec_id: str) -> dict:
    """판례 본문 조회 API 호출"""
    params = {
        "OC": OC,
        "target": "prec",
        "type": "JSON",
        "ID": prec_id,
    }
    resp = session.get(DETAIL_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # 상세 API는 PrecService 키 아래에 데이터가 있음
    return data.get("PrecService", data)


def collect_traffic_precedents() -> list[dict]:
    """교통사고 관련 판례를 수집하여 본문까지 가져온다."""
    keywords = [
        "교통사고", "교통사고 과실", "추돌", "충돌사고",
        "음주운전 사고", "도로교통법 위반", "교통사고처리특례법",
        "손해배상 교통사고", "과실비율", "과실상계",
        "보행자 사고", "횡단보도 사고", "킥보드 사고", "자전거 사고",
        "중앙선 침범", "신호위반 사고", "뺑소니",
        "차량 손해배상", "대물배상", "대인배상",
        "주차 차량 충돌", "정차 차량", "후방추돌",
    ]
    all_details = []
    seen_ids = set()

    for keyword in keywords:
        results = search_precedents(keyword, display=2000, page=1)
        if not results:
            continue
        for item in results:
            prec_id = item.get("판례일련번호")
            if prec_id and prec_id not in seen_ids:
                seen_ids.add(prec_id)
                try:
                    detail = get_precedent_detail(prec_id)
                    all_details.append(detail)
                except requests.exceptions.RequestException as e:
                    print(f"    [!] {prec_id} 조회 실패: {e}")
                time.sleep(0.3)
        print(f"  [{keyword}] {len(results)}건 검색, 누적 {len(all_details)}건")

    print(f"총 {len(all_details)}건 수집 완료")
    return all_details


# ── 2. 텍스트 전처리 ──

def _strip_html(text: str) -> str:
    """HTML 태그 제거"""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def build_document(detail: dict) -> dict | None:
    """판례 상세 정보를 임베딩용 문서로 변환"""
    case_name = _strip_html(detail.get("사건명", ""))
    case_number = _strip_html(detail.get("사건번호", ""))
    court = _strip_html(detail.get("법원명", ""))
    date = _strip_html(detail.get("선고일자", ""))
    verdict_type = _strip_html(detail.get("판결유형", ""))
    issue = _strip_html(detail.get("판시사항", ""))
    summary = _strip_html(detail.get("판결요지", ""))
    content = _strip_html(detail.get("판례내용", ""))
    ref_articles = _strip_html(detail.get("참조조문", ""))
    ref_cases = _strip_html(detail.get("참조판례", ""))

    # 교통사고/과실 관련 판례만 필터링
    all_text = f"{case_name} {issue} {summary} {content}"
    traffic_keywords = ["교통사고", "과실", "추돌", "충돌", "운전", "차량", "도로", "신호", "횡단보도", "보행자", "손해배상"]
    if not any(kw in all_text for kw in traffic_keywords):
        return None

    # 임베딩용 텍스트: 핵심 요약 필드만 (사건명 + 판시사항 + 판결요지)
    # 판례내용은 길고 노이즈가 많아 임베딩에서 제외, 리랭킹에서만 활용
    text = f"{case_name}\n{issue}\n{summary}".strip()

    if len(text) < 20:
        return None

    return {
        "id": detail.get("판례일련번호", ""),
        "text": text,
        "metadata": {
            "사건명": case_name,
            "사건번호": case_number,
            "법원명": court,
            "선고일자": date,
            "판결유형": verdict_type,
            "판시사항": issue,
            "판결요지": summary,
            "판례내용": content,
            "참조조문": ref_articles,
            "참조판례": ref_cases,
        },
    }


# ── 3. 임베딩 & Qdrant 저장 ──

class PrecedentRAG:
    def __init__(self):
        os.environ["TMPDIR"] = "/tmp"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        from FlagEmbedding import BGEM3FlagModel, FlagReranker
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        print("BGE-M3 모델 로딩 중...")
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=["cuda:0"])
        print("리랭커 모델 로딩 중...")
        self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        self.client = QdrantClient(path="./qdrant_data")
        self._ensure_collection(Distance, VectorParams)

    def _ensure_collection(self, Distance, VectorParams):
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            print(f"컬렉션 '{COLLECTION_NAME}' 생성 완료")

    def embed(self, texts: list[str], max_length: int = 512) -> list[list[float]]:
        result = self.model.encode(texts, batch_size=64, max_length=max_length)
        return result["dense_vecs"].tolist()

    def index_documents(self, documents: list[dict], batch_size: int = 64):
        """문서 리스트를 배치 단위로 임베딩하여 Qdrant에 저장"""
        from qdrant_client.models import PointStruct

        total = len(documents)
        upsert_executor = ThreadPoolExecutor(max_workers=1)
        upsert_future = None

        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]
            vectors = self.embed(texts)

            points = [
                PointStruct(
                    id=i + j,
                    vector=vec,
                    payload={
                        "prec_id": doc["id"],
                        "text": doc["text"],
                        **doc["metadata"],
                    },
                )
                for j, (doc, vec) in enumerate(zip(batch, vectors))
            ]

            if upsert_future is not None:
                upsert_future.result()

            upsert_future = upsert_executor.submit(
                self.client.upsert, collection_name=COLLECTION_NAME, points=points
            )
            print(f"  임베딩 진행: {min(i + batch_size, total)}/{total}건")

        if upsert_future is not None:
            upsert_future.result()
        upsert_executor.shutdown()

        print(f"{total}건 인덱싱 완료")

    QUERY_SYNONYMS = {
        "뺑소니": "도주치상 도주차량 사고후미조치 교통사고 후 도주",
        "음주운전": "도로교통법위반 음주측정 혈중알코올농도 위드마크",
        "무면허": "무면허운전 도로교통법위반",
        "과실비율": "과실상계 기여과실 공동과실",
        "과실상계": "과실비율 기여과실 공동과실",
        "보행자": "횡단보도 보행 피해자",
        "추돌": "충돌 다중추돌 접촉사고",
        "사망": "치사 사상 업무상과실치사",
        "상해": "치상 부상 업무상과실치상",
        "신호위반": "신호 도로교통법위반 교차로",
        "손해배상": "위자료 대인배상 대물배상 일실수입",
        "위자료": "손해배상 정신적 손해 위자료 산정",
        "역주행": "중앙선침범 중앙선 역주행",
        "중앙선침범": "역주행 중앙선",
    }

    def _expand_query(self, query: str) -> str:
        """쿼리에 법률 동의어를 추가하여 확장"""
        expansions = []
        for keyword, synonyms in self.QUERY_SYNONYMS.items():
            if keyword in query:
                expansions.append(synonyms)
        if expansions:
            return f"{query} ({' '.join(expansions)})"
        return query

    def search(self, query: str, top_k: int = 5, rerank_k: int = 100) -> list[dict]:
        """쿼리와 유사한 판례 검색 (dense 상위 후보 → cross-encoder 리랭킹)"""
        # 쿼리 확장 (법률 동의어)
        expanded_query = self._expand_query(query)
        if expanded_query != query:
            print(f"쿼리 확장: {expanded_query}")

        # 1단계: dense 벡터로 상위 후보 가져오기 (원본 쿼리 사용)
        query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
        query_vec = self.embed([query_with_instruction])[0]
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=rerank_k,
            with_payload=True,
        )

        if not results.points:
            return []

        print(f"상위 {len(results.points)}건 판례 대상 리랭킹...")

        # 2단계: cross-encoder 리랭킹 (판시사항 + 판결요지 기준)
        rerank_texts = []
        for p in results.points:
            rerank_text = (
                f"{p.payload.get('사건명', '')}\n"
                f"{p.payload.get('판시사항', '')}\n"
                f"{p.payload.get('판결요지', '')}"
            )
            rerank_texts.append(rerank_text)

        sentence_pairs = [[expanded_query, text] for text in rerank_texts]
        rerank_scores = self.reranker.compute_score(sentence_pairs, normalize=True)
        if isinstance(rerank_scores, float):
            rerank_scores = [rerank_scores]

        scored = []
        for point, score in zip(results.points, rerank_scores):
            scored.append({
                "score": score,
                "사건명": point.payload.get("사건명", ""),
                "사건번호": point.payload.get("사건번호", ""),
                "법원명": point.payload.get("법원명", ""),
                "선고일자": point.payload.get("선고일자", ""),
                "판결유형": point.payload.get("판결유형", ""),
                "판시사항": point.payload.get("판시사항", ""),
                "판결요지": point.payload.get("판결요지", ""),
                "판례내용": point.payload.get("판례내용", ""),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        print(f"리랭킹 완료. 상위 {top_k}건 반환.")
        return scored[:top_k]


# ── 4. 실행 ──

def ingest():
    """판례 수집 → 임베딩 → Qdrant 저장"""
    import gc

    cache_path = "ingest_cache.json"

    if os.path.exists(cache_path):
        print(f"=== 캐시 파일({cache_path})에서 로드 ===")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"캐시에서 {len(cached)}건 로드 완료")

        # 캐시가 이미 build_document 처리된 형태({id, text, metadata})인지 확인
        if cached and "metadata" in cached[0]:
            # metadata에서 원본 필드를 꺼내 build_document를 다시 적용
            raw_data = []
            for item in cached:
                detail = {"판례일련번호": item["id"], **item["metadata"]}
                raw_data.append(detail)
        else:
            raw_data = cached
    else:
        print("=== 판례 수집 시작 ===")
        raw_data = collect_traffic_precedents()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False)
        print(f"캐시 저장 완료: {cache_path}")

    documents = []
    for detail in raw_data:
        doc = build_document(detail)
        if doc:
            documents.append(doc)

    print(f"{len(documents)}건 문서 변환 완료")

    # requests 세션 정리 후 모델 로딩
    session.close()
    del raw_data
    gc.collect()

    rag = PrecedentRAG()
    rag.index_documents(documents)
    print("=== 인덱싱 완료 ===")


def query(question: str, top_k: int = 1):
    """질문에 대해 유사 판례 검색"""
    rag = PrecedentRAG()
    results = rag.search(question, top_k=top_k)

    # 유사도 임계값 필터링
    filtered = [r for r in results if r["score"] >= SCORE_THRESHOLD]

    print(f"\n=== '{question}' 검색 결과 ===\n")

    if not filtered:
        print("관련 판례를 찾지 못했습니다.")
        return []

    for i, r in enumerate(filtered, 1):
        print(f"[{i}] (유사도: {r['score']:.4f})")
        print(f"    사건명: {r['사건명']}")
        print(f"    사건번호: {r['사건번호']}")
        print(f"    법원명: {r['법원명']} | 선고일자: {r['선고일자']}")
        print(f"    판결유형: {r['판결유형']}")

        # 판례 본문에서 핵심 정보 추출
        full_text = f"{r['판결요지']} {r['판례내용']}"

        # 과실비율
        ratios = re.findall(r'(?:과실|책임)\s*(?:비율|상계)?\s*[^0-9]*(\d{1,3})\s*[:%：]\s*(\d{1,3})', full_text)
        if not ratios:
            ratios = re.findall(r'(\d{1,3})\s*[%％퍼센트]\s*(?:의?\s*과실|로\s*(?:봄|정함|인정))', full_text)
        if ratios:
            if isinstance(ratios[0], tuple):
                print(f"    ★ 과실비율: {ratios[0][0]}:{ratios[0][1]}")
            else:
                print(f"    ★ 과실비율: {ratios[0]}%")

        # 형량 (징역/벌금/금고)
        sentences = re.findall(r'(징역|금고|벌금)\s*([\d,]+\s*(?:년|월|만\s*원|원)[\s\d,년월일]*)', full_text)
        if sentences:
            for stype, sval in sentences[:2]:
                print(f"    ★ {stype}: {sval.strip()}")

        # 손해배상액
        damages = re.findall(r'(?:손해배상|위자료|합계)\s*(?:금액|액)?[^0-9]*([\d,]+)\s*원', full_text)
        if damages:
            for d in damages[:2]:
                print(f"    ★ 배상액: {d}원")

        print(f"    판시사항: {r['판시사항']}")
        body = r['판결요지'] if r['판결요지'] else r['판례내용']
        print(f"    판결요지: {body}")
        print()

    return filtered


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법:")
        print("  python rag_engine.py ingest          # 판례 수집 및 인덱싱")
        print('  python rag_engine.py query "질문"     # 유사 판례 검색')
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        ingest()
    elif command == "query":
        q = sys.argv[2] if len(sys.argv) > 2 else "교통사고 과실 비율"
        query(q)
    else:
        print(f"알 수 없는 명령: {command}")