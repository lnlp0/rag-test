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
EMBEDDING_DIM = 1024
SCORE_THRESHOLD = 0.0

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))


# ── 태깅/필터 공통 설정 ──
# _auto_tag(인덱싱)과 _detect_filters(검색) 양쪽에서 공유

TAG_RULES = {
    "사고유형": [
        ("추돌",     ["추돌", "후방충돌"]),
        ("충돌",     ["충돌", "접촉"]),
        ("중앙선침범", ["중앙선 침범", "중앙선을 침범", "중앙선을 넘"]),
        ("신호위반",  ["신호위반", "신호를 위반", "적색신호", "적색등화"]),
        ("차선변경",  ["차선변경", "차선을 변경", "차로변경", "차로를 변경"]),
        ("좌회전",   ["좌회전"]),
        ("우회전",   ["우회전"]),
        ("유턴",     ["유턴"]),
        ("앞지르기",  ["앞지르기", "추월"]),
        ("후진",     ["후진"]),
        ("도주",     ["도주", "뺑소니", "미조치"]),
        ("음주",     ["음주", "혈중알코올", "주취"]),
        ("무면허",   ["무면허"]),
        ("과속",     ["과속", "제한속도"]),
        ("졸음",     ["졸음"]),
        ("급발진",   ["급발진", "급가속"]),
    ],
    "차량유형": [
        ("자전거",   ["자전거"]),
        ("킥보드",   ["킥보드", "개인형 이동장치"]),
        ("오토바이",  ["오토바이", "원동기장치자전거", "이륜"]),
        ("화물차",   ["화물차", "화물자동차", "트럭", "덤프", "레미콘"]),
        ("버스",     ["버스"]),
        ("택시",     ["택시"]),
        ("승용차",   ["승용차", "승용자동차"]),
    ],
    "장소": [
        ("횡단보도",  ["횡단보도"]),
        ("교차로",   ["교차로"]),
        ("고속도로",  ["고속도로"]),
        ("터널",     ["터널"]),
        ("주차장",   ["주차장"]),
        ("스쿨존",   ["어린이 보호구역", "어린이보호구역", "스쿨존"]),
    ],
}

PARTY_RULES = [
    ("차대사람", ["보행자", "횡단보도를 건너", "보행 중"]),
    ("차대이륜", ["자전거", "킥보드", "오토바이", "원동기장치자전거", "이륜"]),
]

# traffic_keywords 필터도 TAG_RULES에서 자동 생성
_ALL_TAG_KEYWORDS = set()
for rules in TAG_RULES.values():
    for _, kws in rules:
        _ALL_TAG_KEYWORDS.update(kws)
for _, kws in PARTY_RULES:
    _ALL_TAG_KEYWORDS.update(kws)
TRAFFIC_FILTER_KEYWORDS = sorted(_ALL_TAG_KEYWORDS | {
    "교통사고", "과실", "운전", "차량", "도로", "손해배상",
    "자동차", "보험", "배상", "위자료", "과실상계",
})


# ── 1. 판례 데이터 수집 ──

def search_precedents(query: str, display: int = 20, page: int = 1) -> list[dict]:
    params = {
        "OC": OC, "target": "prec", "type": "JSON",
        "query": query, "display": display, "page": page,
    }
    resp = session.get(SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    prec_list = data.get("PrecSearch", {}).get("prec", [])
    if isinstance(prec_list, dict):
        prec_list = [prec_list]
    return prec_list


def get_precedent_detail(prec_id: str) -> dict:
    params = {"OC": OC, "target": "prec", "type": "JSON", "ID": prec_id}
    resp = session.get(DETAIL_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("PrecService", data)


def search_precedents_all_pages(query: str, display: int = 2000) -> list[dict]:
    all_results = []
    page = 1
    while True:
        results = search_precedents(query, display=display, page=page)
        if not results:
            break
        all_results.extend(results)
        if len(results) < display:
            break
        page += 1
        time.sleep(0.3)
    return all_results


INGEST_KEYWORDS = [
    # 기본
    "교통사고", "교통사고 과실", "교통사고처리특례법",
    "도로교통법 위반", "도로교통법위반",
    # 사고 유형
    "추돌", "충돌사고", "후방추돌", "다중추돌", "연쇄추돌",
    "중앙선 침범", "역주행", "신호위반 사고",
    "차선변경 사고", "유턴 사고", "좌회전 사고", "우회전 사고",
    "끼어들기 사고", "앞지르기 사고",
    "급제동 사고", "급차선변경", "급발진 사고",
    "비보호 좌회전", "불법유턴",
    # 사고 환경/장소
    "터널 사고", "터널 내 충돌",
    "빗길 사고", "결빙 사고", "우천 교통사고",
    "야간 교통사고", "야간 보행자",
    "과속 사고", "제한속도 초과",
    "졸음운전", "졸음운전 사고",
    "공사구간 사고", "갓길 사고", "갓길 주정차",
    # 차량 유형
    "자전거 사고", "킥보드 사고", "이륜차 사고", "오토바이 사고",
    "화물차 사고", "버스 사고", "택시 사고",
    "원동기장치자전거", "개인형 이동장치",
    "전동킥보드 사고", "전기자전거 사고", "전동휠 사고",
    "덤프트럭 사고", "레미콘 사고", "렌터카 사고",
    "이륜자동차", "긴급차량 사고", "구급차 사고",
    # 피해자 유형/장소
    "보행자 사고", "횡단보도 사고", "어린이 교통사고",
    "고속도로 사고", "교차로 사고", "이면도로 사고",
    "주차장 사고", "주차 차량 충돌", "정차 차량",
    "노인 교통사고", "장애인 교통사고",
    "스쿨존 사고", "어린이보호구역 사고",
    # 결과/처벌
    "업무상과실치사", "업무상과실치상",
    "음주운전 사고", "음주운전 치사", "음주운전 치상",
    "무면허운전 사고", "뺑소니", "도주치상", "도주치사",
    "위험운전치사상", "특정범죄가중처벌 도주",
    # 법적 쟁점
    "처벌불원 교통사고", "반의사불벌 교통사고", "합의 교통사고",
    "블랙박스 증거", "위드마크 공식",
    "공동불법행위 교통사고", "운행자 책임",
    "신뢰원칙 교통사고", "주의의무 위반",
    # 민사/보험
    "손해배상 교통사고", "과실비율", "과실상계",
    "차량 손해배상", "대물배상", "대인배상",
    "자동차보험", "자동차손해배상보장법", "책임보험",
    "위자료 교통사고", "일실수입",
]


def collect_traffic_precedents() -> list[dict]:
    all_details = []
    seen_ids = set()

    for keyword in INGEST_KEYWORDS:
        results = search_precedents_all_pages(keyword)
        if not results:
            continue
        new_count = 0
        for item in results:
            prec_id = item.get("판례일련번호")
            if prec_id and prec_id not in seen_ids:
                seen_ids.add(prec_id)
                try:
                    detail = get_precedent_detail(prec_id)
                    all_details.append(detail)
                    new_count += 1
                except requests.exceptions.RequestException as e:
                    print(f"    [!] {prec_id} 조회 실패: {e}")
                time.sleep(0.3)
        print(f"  [{keyword}] {len(results)}건 검색 (신규 {new_count}건), 누적 {len(all_details)}건")

    print(f"총 {len(all_details)}건 수집 완료")
    return all_details


# ── 2. 텍스트 전처리 & 태깅 ──

def _strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _auto_tag(text: str) -> dict:
    """TAG_RULES 기반으로 메타데이터 자동 태깅"""
    tags = {}
    for field, rules in TAG_RULES.items():
        matched = [label for label, kws in rules if any(kw in text for kw in kws)]
        default = "일반도로" if field == "장소" else "기타"
        tags[field] = ",".join(matched) if matched else default

    # 당사자 구분
    tags["당사자"] = "차대차"
    for label, kws in PARTY_RULES:
        if any(kw in text for kw in kws):
            tags["당사자"] = label
            break

    return tags


def _split_points(text: str) -> list[str]:
    """판시사항의 [1], [2] 등 멀티포인트를 분리"""
    parts = re.split(r'\[(\d+)\]\s*', text)
    # parts: ['', '1', '첫번째 쟁점...', '2', '두번째 쟁점...', ...]
    chunks = []
    i = 1
    while i < len(parts) - 1:
        chunk = parts[i + 1].strip()
        if len(chunk) >= 30:  # 너무 짧은 청크는 무시
            chunks.append(chunk)
        i += 2
    # [1] 마커가 없었거나 분리 결과가 없으면 원문 그대로
    return chunks if chunks else [text]


def build_documents(detail: dict) -> list[dict]:
    """판례 1건 → 멀티포인트 청킹으로 N개 문서 반환"""
    fields = {k: _strip_html(detail.get(k, "")) for k in [
        "사건명", "사건번호", "법원명", "선고일자", "판결유형",
        "판시사항", "판결요지", "판례내용", "참조조문", "참조판례",
    ]}

    all_text = " ".join(fields.values())
    if not any(kw in all_text for kw in TRAFFIC_FILTER_KEYWORDS):
        return []

    issue, summary = fields["판시사항"], fields["판결요지"]

    # [개선1] 빈 판례 제거: 판시사항+판결요지 둘 다 없으면 인덱싱 안 함
    if not issue and not summary:
        return []

    tags = _auto_tag(all_text)
    prec_id = detail.get("판례일련번호", "")
    metadata = {**fields, **tags}

    # [개선3] 멀티포인트 청킹: 판시사항의 [1],[2]를 분리
    if issue:
        chunks = _split_points(issue)
    else:
        chunks = [summary[:500]]

    documents = []
    for idx, chunk in enumerate(chunks):
        embed_text = f"{fields['사건명']}\n{chunk}"
        if len(embed_text) < 20:
            continue
        documents.append({
            "id": f"{prec_id}_{idx}" if len(chunks) > 1 else prec_id,
            "text": embed_text,
            "metadata": metadata,
        })

    return documents


# ── 3. 임베딩 & Qdrant 저장 & 검색 ──

# 쿼리 정규화 매핑 (VLM 출력 + 구어체 → 법률 문서 스타일)
QUERY_NORMALIZE = {
    # 차량 유형
    "자전거": "자전거", "킥보드": "개인형 이동장치 전동킥보드",
    "전동킥보드": "개인형 이동장치 전동킥보드",
    "오토바이": "원동기장치자전거 오토바이",
    "이륜차": "원동기장치자전거 이륜차",
    "트럭": "화물차 화물자동차", "화물차": "화물차 화물자동차",
    "덤프트럭": "덤프트럭 화물차", "레미콘": "레미콘 화물차",
    "버스": "버스 대형차량", "택시": "택시 영업용차량",
    "승용차": "자동차 승용차", "SUV": "자동차 승용차", "세단": "자동차 승용차",
    "구급차": "긴급차량 구급차",
    # 장소
    "횡단보도": "횡단보도 보행자", "교차로": "교차로",
    "골목": "이면도로", "고속도로": "고속도로",
    "터널": "터널", "주차장": "주차장", "갓길": "갓길",
    "스쿨존": "어린이보호구역", "어린이보호구역": "어린이보호구역",
    # 사고 유형
    "우회전": "우회전", "좌회전": "좌회전",
    "비보호 좌회전": "비보호 좌회전", "유턴": "유턴",
    "차선변경": "차선변경 차로변경", "차로변경": "차선변경 차로변경",
    "끼어들기": "끼어들기 차선변경", "앞지르기": "앞지르기 추월",
    "추돌": "추돌 충돌", "충돌": "충돌", "접촉": "접촉 충돌",
    "후진": "후진", "급제동": "급제동 급정거", "급정거": "급제동 급정거",
    "급차선변경": "급차선변경 차선변경", "급발진": "급발진 급가속",
    "중앙선": "중앙선 침범", "역주행": "역주행 중앙선 침범",
    "중앙분리대": "중앙분리대 중앙선",
    "신호": "신호위반 적색등화", "적색": "적색신호 신호위반 적색등화",
    # 환경
    "빗길": "빗길 우천", "결빙": "결빙 빙판",
    "야간": "야간 무등화", "졸음": "졸음운전", "과속": "과속 제한속도",
    # 결과
    "골절": "상해 치상", "사망": "치사 사상 업무상과실치사",
    "상해": "상해 치상 업무상과실치상",
    "대파": "손괴 파손", "파손": "손괴 파손",
    "전도": "전도 전복", "전복": "전복", "에어백": "충돌 사고",
    # 법률 용어
    "뺑소니": "도주치상 도주차량 사고후미조치",
    "도주": "도주 뺑소니 사고후미조치",
    "음주운전": "도로교통법위반 음주측정 혈중알코올농도",
    "무면허": "무면허운전 도로교통법위반",
    "과실비율": "과실상계 기여과실 공동과실",
    "손해배상": "위자료 대인배상 대물배상",
}


class PrecedentRAG:
    def __init__(self):
        os.environ["TMPDIR"] = "/tmp"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        from FlagEmbedding import BGEM3FlagModel, FlagReranker
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, SparseVectorParams, SparseIndexParams,
            PointStruct, SparseVector, Filter, FieldCondition, MatchValue,
            FusionQuery, Fusion, Prefetch,
        )
        # qdrant models를 인스턴스에 보관 (search/index에서 재사용)
        self._qm = type("QM", (), {
            "PointStruct": PointStruct, "SparseVector": SparseVector,
            "Filter": Filter, "FieldCondition": FieldCondition,
            "MatchValue": MatchValue, "FusionQuery": FusionQuery,
            "Fusion": Fusion, "Prefetch": Prefetch,
        })()

        print("BGE-M3 모델 로딩 중...")
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=["cuda:0"])
        print("리랭커 모델 로딩 중...")
        self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        self.client = QdrantClient(path="./qdrant_data")

        # 컬렉션 생성 (없을 때만)
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                },
            )
            print(f"컬렉션 '{COLLECTION_NAME}' 생성 완료 (dense + sparse)")

    def embed(self, texts: list[str], max_length: int = 512) -> dict:
        result = self.model.encode(
            texts, batch_size=64, max_length=max_length,
            return_dense=True, return_sparse=True,
        )
        return {
            "dense": result["dense_vecs"].tolist(),
            "sparse": result["lexical_weights"],
        }

    def index_documents(self, documents: list[dict], batch_size: int = 64):
        qm = self._qm
        total = len(documents)
        upsert_executor = ThreadPoolExecutor(max_workers=1)
        upsert_future = None

        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            embeddings = self.embed([doc["text"] for doc in batch])

            points = []
            for j, (doc, dense_vec, sparse_dict) in enumerate(
                zip(batch, embeddings["dense"], embeddings["sparse"])
            ):
                points.append(qm.PointStruct(
                    id=i + j,
                    vector={
                        "": dense_vec,
                        "sparse": qm.SparseVector(
                            indices=[int(k) for k in sparse_dict.keys()],
                            values=[float(v) for v in sparse_dict.values()],
                        ),
                    },
                    payload={"prec_id": doc["id"], "text": doc["text"], **doc["metadata"]},
                ))

            if upsert_future is not None:
                upsert_future.result()
            upsert_future = upsert_executor.submit(
                self.client.upsert, collection_name=COLLECTION_NAME, points=points,
            )
            print(f"  임베딩 진행: {min(i + batch_size, total)}/{total}건")

        if upsert_future is not None:
            upsert_future.result()
        upsert_executor.shutdown()
        print(f"{total}건 인덱싱 완료 (dense + sparse)")

    # ── 검색 ──

    @staticmethod
    def _normalize_query(query: str) -> str:
        keywords = [terms for trigger, terms in QUERY_NORMALIZE.items() if trigger in query]
        if not keywords:
            return query
        normalized = f"교통사고처리특례법위반 {' '.join(keywords)}"
        print(f"쿼리 정규화: {query}")
        print(f"  → 임베딩용: {normalized}")
        return normalized

    def _detect_filters(self, query: str):
        """[개선5] 쿼리에서 메타데이터 필터 추출 (당사자=must, 장소=should)"""
        qm = self._qm
        must_conditions = []
        should_conditions = []

        # 당사자 필터 (must: 차대차 쿼리에 차대사람 판례가 올라오면 안 됨)
        for label, kws in PARTY_RULES:
            if any(kw in query for kw in kws):
                must_conditions.append(qm.FieldCondition(key="당사자", match=qm.MatchValue(value=label)))
                break

        # 장소 필터 (should: 보조적으로 사용)
        for label, kws in TAG_RULES["장소"]:
            if any(kw in query for kw in kws):
                should_conditions.append(qm.FieldCondition(key="장소", match=qm.MatchValue(value=label)))
                break

        if not must_conditions and not should_conditions:
            return None

        kwargs = {}
        if must_conditions:
            kwargs["must"] = must_conditions
        if should_conditions:
            kwargs["should"] = should_conditions
        return qm.Filter(**kwargs)

    def search(self, query: str, top_k: int = 5, rerank_k: int = 100) -> list[dict]:
        qm = self._qm
        normalized = self._normalize_query(query)

        # dense + sparse 임베딩
        emb = self.embed([normalized])
        dense_vec = emb["dense"][0]
        sp = emb["sparse"][0]
        sparse_vec = qm.SparseVector(
            indices=[int(k) for k in sp.keys()],
            values=[float(v) for v in sp.values()],
        )

        # [개선5] 메타데이터 필터
        query_filter = self._detect_filters(query)
        if query_filter:
            print(f"메타데이터 필터 적용")

        # 하이브리드 검색: dense + sparse → RRF
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qm.Prefetch(query=dense_vec, using="", limit=rerank_k, filter=query_filter),
                qm.Prefetch(query=sparse_vec, using="sparse", limit=rerank_k, filter=query_filter),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=rerank_k,
            with_payload=True,
        )

        # 필터 결과가 너무 적으면 필터 없이 재검색 (fallback)
        if len(results.points) < 10 and query_filter:
            print(f"필터 결과 부족 ({len(results.points)}건), 필터 없이 재검색...")
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    qm.Prefetch(query=dense_vec, using="", limit=rerank_k),
                    qm.Prefetch(query=sparse_vec, using="sparse", limit=rerank_k),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=rerank_k,
                with_payload=True,
            )

        if not results.points:
            return []

        print(f"하이브리드 검색 상위 {len(results.points)}건 → 리랭킹...")

        # [개선2] 리랭킹 텍스트: 판시사항이 짧으면 판결요지도 포함
        rerank_texts = []
        for p in results.points:
            issue = p.payload.get("판시사항", "")
            if len(issue) >= 100:
                rerank_texts.append(f"{p.payload.get('사건명', '')}\n{issue}")
            else:
                summary = p.payload.get("판결요지", "")[:300]
                rerank_texts.append(f"{p.payload.get('사건명', '')}\n{issue}\n{summary}")

        # [개선4] ColBERT reranking: BGE-M3 compute_score (dense+sparse+colbert)
        sentence_pairs = [[query, text] for text in rerank_texts]
        colbert_scores = self.model.compute_score(
            sentence_pairs,
            weights_for_different_modes=[0.2, 0.2, 0.6],  # [dense, sparse, colbert]
        )
        if isinstance(colbert_scores, (float, int)):
            colbert_scores = [colbert_scores]
        # compute_score returns dict with 'colbert+sparse+dense' key or list
        if isinstance(colbert_scores, dict):
            colbert_scores = colbert_scores.get("colbert+sparse+dense", [0.0])

        # cross-encoder도 함께 사용 (앙상블)
        ce_scores = self.reranker.compute_score(sentence_pairs, normalize=True)
        if isinstance(ce_scores, float):
            ce_scores = [ce_scores]

        scored = []
        seen_cases = set()  # 멀티포인트 청킹 중복 제거용
        for point, col_score, ce_score in zip(results.points, colbert_scores, ce_scores):
            # 같은 사건번호 중복 제거 (멀티포인트 청킹)
            case_num = point.payload.get("사건번호", "")
            if case_num in seen_cases:
                continue
            seen_cases.add(case_num)

            # 앙상블: ColBERT 0.4 + cross-encoder 0.6
            rank_score = 0.4 * float(col_score) + 0.6 * float(ce_score)
            display_score = 0.7 + 0.3 * float(ce_score)

            scored.append({
                "score": display_score,
                "_rank_score": rank_score,
                **{k: point.payload.get(k, "") for k in [
                    "사건명", "사건번호", "법원명", "선고일자",
                    "판결유형", "판시사항", "판결요지", "판례내용",
                ]},
            })

        scored.sort(key=lambda x: x["_rank_score"], reverse=True)
        print(f"리랭킹 완료. 상위 {top_k}건 반환.")
        return scored[:top_k]


# ── 4. 실행 ──

def ingest():
    import gc

    cache_path = "ingest_cache.json"

    if os.path.exists(cache_path):
        print(f"=== 캐시 파일({cache_path})에서 로드 ===")
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        # 캐시가 build_document 처리된 형태면 원본으로 복원
        if raw_data and "metadata" in raw_data[0]:
            raw_data = [{"판례일련번호": d["id"], **d["metadata"]} for d in raw_data]
        print(f"캐시에서 {len(raw_data)}건 로드 완료")
    else:
        print("=== 판례 수집 시작 ===")
        raw_data = collect_traffic_precedents()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False)
        print(f"캐시 저장 완료: {cache_path}")

    documents = [doc for detail in raw_data for doc in build_documents(detail)]
    print(f"{len(documents)}건 문서 변환 완료")

    session.close()
    del raw_data
    gc.collect()

    rag = PrecedentRAG()
    rag.index_documents(documents)
    print("=== 인덱싱 완료 ===")


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\[\d+\]\s*', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def query(question: str, top_k: int = 1):
    rag = PrecedentRAG()
    results = rag.search(question, top_k=top_k)
    filtered = [r for r in results if r["score"] >= SCORE_THRESHOLD]

    print(f"\n검색: {question}\n")

    if not filtered:
        print("관련 판례를 찾지 못했습니다.\n")
        return []

    for i, r in enumerate(filtered, 1):
        print(f"[{i}] (유사도: {r['score']:.4f})")
        print(f"    사건명: {r['사건명']}")
        print(f"    사건번호: {r['사건번호']}")
        print(f"    법원명: {r['법원명']} | 선고일자: {r['선고일자']}")
        print(f"    판결유형: {r['판결유형']}")
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
