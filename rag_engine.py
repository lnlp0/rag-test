import os
import re
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# ── 설정 ──
OC = os.getenv("OC")
SEARCH_URL = "http://www.law.go.kr/DRF/lawSearch.do"
DETAIL_URL = "http://www.law.go.kr/DRF/lawService.do"
COLLECTION_NAME = "traffic_precedents"
EMBEDDING_DIM = 1024  # BGE-M3 default dimension
SCORE_THRESHOLD = 0.0  # 임계값 (0이면 항상 결과 반환)

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
        "교통사고", "교통사고 과실", "추돌",
        "음주운전 사고", "도로교통법 위반", "교통사고처리특례법",
        "손해배상 교통사고",
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

    # 임베딩에 사용할 텍스트 구성 (판례내용도 포함하여 검색 품질 향상)
    body = summary if summary else content[:2000]
    text = f"사건명: {case_name}\n판시사항: {issue}\n판결요지: {body}"

    if not text.strip() or len(text) < 20:
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
        print("BGE-M3 모델 로딩 중...")
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self.client = QdrantClient(path="./qdrant_data")
        self._ensure_collection()

    def _ensure_collection(self):
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self.model.encode(texts, batch_size=64, max_length=512)
        return result["dense_vecs"].tolist()

    def index_documents(self, documents: list[dict], batch_size: int = 64):
        """문서 리스트를 배치 단위로 임베딩하여 Qdrant에 저장"""
        total = len(documents)
        idx_offset = 0

        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]
            vectors = self.embed(texts)

            points = [
                PointStruct(
                    id=idx_offset + j,
                    vector=vec,
                    payload={
                        "prec_id": doc["id"],
                        "text": doc["text"],
                        **doc["metadata"],
                    },
                )
                for j, (doc, vec) in enumerate(zip(batch, vectors))
            ]

            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            idx_offset += len(batch)
            print(f"  임베딩 진행: {min(i + batch_size, total)}/{total}건")

        print(f"{total}건 인덱싱 완료")

    def search(self, query: str, top_k: int = 5, candidate_k: int = 100) -> list[dict]:
        """쿼리와 유사한 판례 검색 (dense 상위 후보 → 리랭킹)"""
        # 1단계: dense 벡터로 상위 candidate_k건 가져오기
        query_vec = self.embed([query])[0]
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=candidate_k,
            with_payload=True,
        )

        if not results.points:
            return []

        print(f"상위 {len(results.points)}건 후보 대상 리랭킹 시작...")

        # 리랭킹용 텍스트: 판례내용(사실관계)을 포함하여 구성
        candidate_texts = []
        for p in results.points:
            content = p.payload.get("판례내용", "")[:1500]
            rerank_text = (
                f"사건명: {p.payload.get('사건명', '')}\n"
                f"판시사항: {p.payload.get('판시사항', '')}\n"
                f"판결요지: {p.payload.get('판결요지', '')}\n"
                f"판례내용: {content}"
            )
            candidate_texts.append(rerank_text)

        # 2단계: BGE-M3 compute_score로 리랭킹
        sentence_pairs = [[query, text] for text in candidate_texts]
        rerank_result = self.model.compute_score(
            sentence_pairs,
            batch_size=32,
            max_query_length=512,
            max_passage_length=8192,
            weights_for_different_modes=[0.2, 0.4, 0.4],  # [dense, sparse, colbert]
        )
        rerank_scores = rerank_result["colbert+sparse+dense"]
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
    print("=== 판례 수집 시작 ===")
    raw_data = collect_traffic_precedents()

    documents = []
    for detail in raw_data:
        doc = build_document(detail)
        if doc:
            documents.append(doc)

    print(f"{len(documents)}건 문서 변환 완료")

    rag = PrecedentRAG()
    rag.index_documents(documents)
    print("=== 인덱싱 완료 ===")


def query(question: str, top_k: int = 5):
    """질문에 대해 유사 판례 검색"""
    rag = PrecedentRAG()
    results = rag.search(question, top_k=top_k)

    # 유사도 임계값 필터링
    filtered = [r for r in results if r["score"] >= SCORE_THRESHOLD]

    print(f"\n=== '{question}' 검색 결과 ===\n")

    if not filtered:
        print("관련 판례를 찾지 못했습니다.")
        if results:
            print(f"(최고 유사도: {results[0]['score']:.4f}, 임계값: {SCORE_THRESHOLD})")
        print()
        return []

    for i, r in enumerate(filtered, 1):
        print(f"[{i}] (유사도: {r['score']:.4f})")
        print(f"    사건명: {r['사건명']}")
        print(f"    사건번호: {r['사건번호']}")
        print(f"    법원명: {r['법원명']} | 선고일자: {r['선고일자']}")
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
