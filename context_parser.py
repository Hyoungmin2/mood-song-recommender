"""
자유 텍스트(기분/걷는 속도/날씨 등) -> 오디오 특징 목표값 변환기

기본은 규칙 기반(rule-based) 키워드 매칭: 사람이 직접 정한 키워드-특징
매핑 사전(KEYWORD_RULES)을 문장에 그대로 적용한다.

--- 2026-08-31 추가: 임베딩 기반 의미 매칭 (stage4-2) ---
규칙 기반 매칭은 키워드가 문장에 "정확히" 포함돼야만 잡힌다. 그래서
"장맛비라 꿀꿀하다" 처럼 사전에 없는 표현(paraphrase)은 "비"/"칙칙" 같은
키워드가 하나도 안 걸려서 그냥 데이터셋 평균으로 추천돼버림. 이걸 보완하려고
sentence-transformers 사전학습 다국어 임베딩으로 "문장이 의미적으로 어떤
개념과 제일 비슷한지"를 추가로 비교하는 2단계를 넣음:

  1단계(기존, 항상 동작): 키워드가 문장에 그대로 들어있으면 매칭 -> 기존과
     100% 동일한 동작/결과를 보장(회귀 없음).
  2단계(신규, 선택적): 1단계에서 못 잡은 개념들에 대해서만, 문장 임베딩과
     "그 개념을 나타내는 예문들"의 코사인 유사도를 비교해서
     EMBEDDING_SIMILARITY_THRESHOLD 이상이면 매칭.

sentence-transformers가 설치 안 돼있으면(ImportError) 2단계는 조용히
건너뛰고 1단계 결과만 반환한다 - 즉 이 라이브러리가 없어도 기존 동작이
그대로 보장됨.

### 2026-08-31 수정: 키워드 자체가 아니라 "예문 문장"과 비교하도록 변경

처음엔 문장 임베딩을 키워드 원문("신나", "비" 같은 2글자 단어) 임베딩과
직접 비교했는데, min이 로컬에서 calibrate_embedding_threshold.py를 돌려본
결과 완전히 무관한 문장("오늘 저녁 뭐 먹을지 고민중이야")도 "신나"와 유사도
0.93이 나오는 등, 관련 없는 문장이 실제 매칭 문장보다 유사도가 더 높게
나오는 경우가 있어서 어떤 threshold를 잡아도 구분이 안 되는 문제를 발견함.
원인은 SBERT 계열 모델이 "문장 대 문장" 의미 비교에 최적화돼 있어서, 문맥이
없는 짧은 단어 하나를 비교 대상으로 쓰면 임베딩이 한쪽으로 쏠려서(비등방성/
hubness) 아무 문장과도 유사도가 높게 나오는 경향이 있기 때문으로 판단함.
그래서 각 개념(동의어 그룹)마다 자연스러운 예문 몇 개(`CONCEPT_EXAMPLE_SENTENCES`)를
직접 써두고, 그 예문들과 비교하도록 바꿈 - 이게 SBERT를 의도대로 쓰는 방식.

### 2026-08-31 수정 2: 모델을 한국어 전용 SBERT로 교체

예문 방식으로 바꾼 뒤 min이 다시 로컬에서 돌려봤는데, 이번엔 무관한 문장
("이 코드 왜 안 돌아가지" 등)도 여전히 대부분의 개념과 유사도 0.93~0.97로
나오는 문제가 계속됨 - 실제 관련 있는 문장(0.6~0.9대)과 명확히 구분이 안 됨.
(테스트 케이스 중 일부는 예문과 거의 똑같은 문장을 써서 1.000이 나온 것도
있었는데 그건 내가 테스트 문장과 예문을 겹치게 짠 실수였고, 그거 빼고 봐도
문제가 여전했음.) `paraphrase-multilingual-MiniLM-L12-v2`는 50개 언어를 한
모델로 커버하는 범용 모델이라 한국어 학습 데이터 비중이 상대적으로 얕고,
그 결과 한국어 문장끼리는 의미와 무관하게 임베딩이 한쪽으로 쏠려서(전반적으로
코사인 유사도가 다 높게 나오는) 것으로 판단함. 그래서 한국어 STS(문장 의미
유사도) 데이터로 직접 학습/검증된 `jhgan/ko-sroberta-multitask`(KorSTS
Spearman 85.6, KorNLI+KorSTS로 멀티태스크 학습, 한국어 NLP 커뮤니티에서
문장 유사도 작업에 흔히 쓰이는 모델)로 교체함. 이것도 아직 실제로 테스트
못 해봄 - 로컬에서 재검증 필요.
"""
from build_dataset import FEATURE_COLS

# 각 키워드가 매칭되면 어떤 특징을 어느 방향으로 얼마나 조정할지 정의.
# 값은 0~1 범위(loudness, tempo는 별도 스케일)이고, 부호가 +면 그 특징을 높이고
# -면 낮추는 식으로 베이스라인(평균)에서 가감함.
KEYWORD_RULES = {
    # 기분
    "신남": {"energy": +0.3, "valence": +0.3, "danceability": +0.2},
    "신나": {"energy": +0.3, "valence": +0.3, "danceability": +0.2},
    "행복": {"energy": +0.2, "valence": +0.3},
    "happy": {"energy": +0.2, "valence": +0.3},
    "우울": {"energy": -0.2, "valence": -0.3},
    "슬픔": {"energy": -0.2, "valence": -0.3},
    "슬퍼": {"energy": -0.2, "valence": -0.3},
    "칙칙": {"energy": -0.2, "valence": -0.3},
    "다운": {"energy": -0.2, "valence": -0.2},
    "차분": {"energy": -0.3, "acousticness": +0.2},
    "평온": {"energy": -0.3, "acousticness": +0.2},
    "잔잔": {"energy": -0.3, "acousticness": +0.2},

    # 걷는 속도
    "느리게": {"tempo": -30},
    "느림": {"tempo": -30},
    "산책": {"tempo": -20},
    "여유": {"tempo": -20},
    "빠르게": {"tempo": +30},
    "빠름": {"tempo": +30},
    "뛰면서": {"tempo": +40, "energy": +0.2},
    "조깅": {"tempo": +40, "energy": +0.2},
    "런닝": {"tempo": +40, "energy": +0.2},

    # 날씨
    "맑음": {"valence": +0.2, "acousticness": -0.1},
    "화창": {"valence": +0.2, "acousticness": -0.1},
    "비": {"acousticness": +0.2, "valence": -0.15},
    "장마": {"acousticness": +0.2, "valence": -0.2},
    "흐림": {"valence": -0.1},
    "구름": {"valence": -0.05},
    # 2026-08-31 추가: "시원한 바람이 불어"가 등록된 키워드가 하나도 없어서
    # 임베딩 2단계로 넘어갔다가 "맑음"(날씨가 맑아, 유사도 0.53)으로 잘못
    # 매칭되는 걸 발견함(ISSUES.md 참고) - "바람"과 "맑음"은 다른 개념인데
    # 사전학습 모델이 둘 다 "날씨 얘기"로 뭉뚱그려서 비슷하게 본 것으로 보임.
    # 임베딩의 판단 기준 자체는 못 고치지만, 이 표현을 1단계 규칙으로
    # 직접 등록하면 애매한 임베딩 추측에 기대지 않고 정확히 잡을 수 있음.
    "바람": {"valence": +0.15, "acousticness": +0.1, "energy": -0.1},
    "시원": {"valence": +0.15, "acousticness": +0.1, "energy": -0.1},
}

# 개념(동의어 그룹)별 자연스러운 예문. 임베딩 비교는 키워드 원문이 아니라
# 이 예문들과 함. 그룹에 속한 키워드 중 하나라도 이 dict에 있으면 그 예문을
# 쓰고, 없으면 키워드 자체를 예문 취급(폴백 - 대부분 커버되도록 다 채워둠).
# 2026-08-31 수정: TEST_CASES(calibrate_embedding_threshold.py)와 겹치거나
# 거의 동일한 예문을 실수로 넣었던 걸 정리함(공정한 재검증을 위해 - 예문과
# 테스트 문장이 똑같으면 "매칭 성공"이 그냥 문자열 일치일 뿐 진짜 의미
# 매칭 검증이 안 됨).
CONCEPT_EXAMPLE_SENTENCES = {
    "신남": ["완전 신난다", "기분이 들떠", "신나는 하루야", "너무 흥분돼"],
    "행복": ["오늘 정말 행복해", "기분이 너무 좋아", "요즘 만족스러워"],
    "우울": ["기분이 우울해", "마음이 무겁고 힘들어", "울적한 기분이야"],
    "다운": ["기분이 다운됐어", "아무것도 하기 싫어", "무기력한 느낌이야"],
    "차분": ["마음이 차분해", "평온한 느낌이야", "고요하게 있고 싶어"],
    "느리게": ["천천히 걷고 싶어", "느긋하게 있고 싶어"],
    "산책": ["여유롭게 밖을 거니는 중이야", "동네를 어슬렁어슬렁 걷고 있어"],
    "빠르게": ["빠르게 움직이고 싶어", "속도감 있게 가고 싶어"],
    "뛰면서": ["지금 뛰고 있어", "조깅하는 중이야", "밖에서 달리기 하는 중이야"],
    "맑음": ["날씨가 맑아", "구름 한 점 없이 화창해"],
    "비": ["비가 오고 있어", "밖에 비 내려", "비가 와서 기분이 가라앉아"],
    "장마": ["장마철이라 계속 비만 와서 우중충해"],
    "흐림": ["날씨가 흐려", "하늘이 잔뜩 흐리다"],
    "구름": ["하늘에 구름이 잔뜩 꼈어"],
    "바람": ["선선한 바람이 불어와", "바람이 산들산들 불어서 기분 좋아", "시원한 바람 덕분에 상쾌해"],
}

# 문장이 이 유사도 이상이면 "의미적으로 비슷하다"고 판단.
# 2026-08-31: jhgan/ko-sroberta-multitask + calibrate_embedding_threshold.py로
# min이 로컬에서 실측 검증한 값. 무관한 문장 최고 유사도 0.345, 실제 매칭
# 문장 최저 유사도 0.548 - 그 사이 중간값인 0.45로 설정(양쪽 다 여유
# 있게 분리됨). ISSUES.md stage4-2e 참고.
EMBEDDING_SIMILARITY_THRESHOLD = 0.45
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

_embedder = None
_concept_groups = None       # [(동의어 키워드 리스트, delta dict, 예문 리스트), ...]
_concept_embeddings = None   # concept_groups와 같은 순서의 예문 임베딩 텐서 리스트


def _build_concept_groups():
    """
    delta가 완전히 같은 키워드들("신남"/"신나"처럼 진짜 동의어)을 하나의
    개념으로 묶는다. 안 묶으면 임베딩 매칭에서 동의어 여러 개가 동시에
    threshold를 넘겨 같은 delta가 중복으로 더해질 수 있음 - 개념당 최댓값
    유사도 1번만 적용되도록 하기 위한 전처리.
    """
    groups = {}
    for keyword, adjustments in KEYWORD_RULES.items():
        key = tuple(sorted(adjustments.items()))
        groups.setdefault(key, []).append(keyword)

    result = []
    for key, keywords in groups.items():
        examples = None
        for kw in keywords:
            if kw in CONCEPT_EXAMPLE_SENTENCES:
                examples = CONCEPT_EXAMPLE_SENTENCES[kw]
                break
        if not examples:
            examples = keywords  # 예문 없으면 키워드 원문이라도 폴백
        result.append((keywords, dict(key), examples))
    return result


def _get_concept_groups():
    global _concept_groups
    if _concept_groups is None:
        _concept_groups = _build_concept_groups()
    return _concept_groups


def _get_embedder():
    """
    sentence-transformers 모델을 최초 1회만 로드해서 프로세스 안에 캐싱.
    설치 안 돼있으면 ImportError를 그대로 올려서 호출부(parse_context)가
    잡아 처리하게 함.
    """
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def _get_concept_embeddings():
    """
    각 개념(동의어 그룹)의 예문들을 미리 임베딩해서 캐싱. 문장이 들어올
    때마다 예문 전체를 다시 임베딩하면 느려지므로, 이 부분만 최초 1회
    계산하고 이후엔 입력 문장 임베딩만 매번 새로 계산함.
    """
    global _concept_embeddings
    if _concept_embeddings is None:
        embedder = _get_embedder()
        groups = _get_concept_groups()
        _concept_embeddings = [
            embedder.encode(examples, convert_to_tensor=True) for _, _, examples in groups
        ]
    return _concept_embeddings


def _semantic_matches(text, exclude_keywords, lang="ko"):
    """
    1단계(substring)에서 이미 잡힌 키워드는 제외하고, 나머지 개념들에 대해
    문장 임베딩 <-> 예문 임베딩 코사인 유사도를 비교한다. 개념별로 예문 중
    최댓값 유사도가 threshold를 넘으면 매칭으로 판단.
    반환: [(표시용 라벨, delta dict), ...]

    lang="en"이면 라벨의 안내 문구("의미유사"/"예문:")만 영어로 바꾸고,
    실제로 매칭된 키워드/예문 자체는 그대로 한국어로 남김 - 이건 번역이
    아니라 "입력 문장에서 실제로 이 한국어 표현이 감지됐다"는 사실을
    보여주는 거라, 매칭된 원문(한국어)을 영어로 바꾸면 오히려 무슨 근거로
    매칭됐는지 알 수 없게 됨.
    """
    from sentence_transformers import util

    embedder = _get_embedder()
    groups = _get_concept_groups()
    group_embeddings = _get_concept_embeddings()

    text_emb = embedder.encode(text, convert_to_tensor=True)
    results = []
    for (keywords, adjustments, examples), ex_embs in zip(groups, group_embeddings):
        # 이미 substring으로 잡힌 키워드가 하나라도 있는 그룹은 건너뜀
        # (중복 적용 방지 - 1단계에서 이미 delta를 더했음)
        if any(k in exclude_keywords for k in keywords):
            continue
        sims = util.cos_sim(text_emb, ex_embs)[0]
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        if best_sim >= EMBEDDING_SIMILARITY_THRESHOLD:
            if lang == "en":
                label = f"{keywords[0]} (semantic match {best_sim:.2f}, example: '{examples[best_idx]}')"
            else:
                label = f"{keywords[0]}(의미유사 {best_sim:.2f}, 예문:'{examples[best_idx]}')"
            results.append((label, adjustments))
    return results


def parse_context(text, df, use_embedding=True, lang="ko"):
    """
    문장에서 키워드를 찾아 데이터셋 평균 기준으로 조정된 목표 특징 벡터를 만든다.
    df: 오디오 특징 평균을 계산할 기준 데이터셋 (보통 stage2_dataset.csv)
    use_embedding: True면 1단계(substring) 이후 2단계(임베딩 의미 매칭)도
        시도한다. sentence-transformers가 설치 안 돼있으면 자동으로 1단계
        결과만 반환(에러 없이 조용히 폴백 - 기존 동작 그대로 보존).
    lang: "en"이면 임베딩 매칭 라벨의 안내 문구만 영어로 바꿈(2026-08-31
        영어 UI 토글 추가 - _semantic_matches 참고). 1단계 substring
        매칭은 원래 키워드를 그대로 보여주므로 lang과 무관하게 항상 동일.
    반환: dict {feature_name: target_value}, 그리고 매칭된 키워드 리스트
        (임베딩으로 매칭된 항목은 "키워드(의미유사 0.xx, 예문:'...')" 형식으로 구분 표시)
    """
    baseline = df[FEATURE_COLS].mean().to_dict()
    target = baseline.copy()
    matched = []
    matched_raw_keywords = set()

    # 1단계: 기존 substring 매칭
    # 2026-08-31 수정: "바람"/"시원"처럼 delta가 완전히 같은 동의어
    # 키워드 두 개가 한 문장에 동시에 등장하면(예: "시원한 바람이 불어")
    # 델타가 중복으로 두 번 더해지는 문제를 발견함(ISSUES.md 참고).
    # 2단계(임베딩)는 처음부터 _build_concept_groups()로 동의어를 묶어서
    # 델타를 1번만 적용했는데, 1단계는 그런 보호 장치가 없었음. 표시용
    # matched 리스트에는 실제로 매칭된 키워드를 그대로 다 보여주되(둘 다
    # 문장에 있었다는 사실 자체는 유효한 정보라 유지함), 델타 적용은
    # 같은 개념(동일 delta dict)당 한 번만 하도록 고침.
    applied_concept_keys = set()
    for keyword, adjustments in KEYWORD_RULES.items():
        if keyword in text:
            matched.append(keyword)
            matched_raw_keywords.add(keyword)
            concept_key = tuple(sorted(adjustments.items()))
            if concept_key not in applied_concept_keys:
                applied_concept_keys.add(concept_key)
                for feature, delta in adjustments.items():
                    target[feature] = target.get(feature, 0) + delta

    # 2단계: 임베딩 의미 매칭 (선택적, 라이브러리 없으면 자동 스킵)
    if use_embedding:
        try:
            semantic_hits = _semantic_matches(text, matched_raw_keywords, lang=lang)
        except ImportError:
            semantic_hits = []
        for label, adjustments in semantic_hits:
            matched.append(label)
            for feature, delta in adjustments.items():
                target[feature] = target.get(feature, 0) + delta

    # 0~1 범위 특징들은 범위를 벗어나지 않게 clip (tempo, loudness는 범위가 다르므로 제외)
    bounded_features = [
        "danceability", "energy", "valence",
        "acousticness", "instrumentalness", "liveness", "speechiness",
    ]
    for f in bounded_features:
        if f in target:
            target[f] = max(0.0, min(1.0, target[f]))

    return target, matched


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/stage2_dataset.csv")
    target, matched = parse_context("오늘 비오고 기분 칙칙해", df)
    print("매칭된 키워드:", matched)
    print("목표 특징값:", target)
