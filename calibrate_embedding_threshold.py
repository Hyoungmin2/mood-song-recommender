"""
임베딩 기반 의미 매칭(context_parser.py stage4-2) threshold 보정용 스크립트.

개발 중 쓰던 원격 실행 환경(클라우드 실행 환경 / "내 컴퓨터에서 실행" 연동 VM)
양쪽 다 huggingface.co 접근이 막혀있어서 실제 모델로 테스트를 못 해봤음
(ISSUES.md의 stage4-2b 항목 참고). 그래서 EMBEDDING_SIMILARITY_THRESHOLD = 0.5는 검증 안 된 잠정값임.

--- 2026-08-31 수정 1 ---
1차 버전은 문장을 키워드 원문("신나" 같은 2글자 단어)과 직접 비교했는데,
min이 로컬에서 돌려본 결과 완전 무관한 문장도 "신나"와 유사도 0.93이 나오는
등 threshold로 구분이 안 되는 문제가 있었음. context_parser.py를 예문 문장
(CONCEPT_EXAMPLE_SENTENCES)과 비교하는 방식으로 고쳤음.

--- 2026-08-31 수정 2 ---
예문 방식으로도 무관한 문장이 여전히 대부분 0.9대로 나오는 문제가 계속됨.
알고 보니 (a) 테스트 문장 일부가 예문과 실수로 거의 동일해서 그 항목은
애초에 공정한 테스트가 아니었고, (b) 그걸 제외하고 봐도 모델
(`paraphrase-multilingual-MiniLM-L12-v2`)이 한국어에서 전반적으로 코사인
유사도를 높게 뭉뚱그려 내는 경향이 있었음. `context_parser.py`의 예문을
테스트 문장과 안 겹치게 정리하고, 모델을 한국어 STS 전용으로 학습된
`jhgan/ko-sroberta-multitask`로 교체함. 이 버전으로 다시 로컬에서 돌려서
결과 확인 필요 (첫 실행 시 새 모델을 다시 다운로드하므로 시간 걸림).

실행 전 준비 (이미 했으면 생략):
    cd stage2
    pip3 install -r requirements-embedding.txt

실행:
    python3 calibrate_embedding_threshold.py
"""
from context_parser import _build_concept_groups, _get_embedder

# 사전에 있는 표현이 "그대로" 들어있지 않은 paraphrase 테스트 문장들.
# (기대하는 개념, 문장) 쌍 - 기대값은 사람이 미리 판단한 정답(육안 검수용).
TEST_CASES = [
    ("신남/행복 계열", "완전 럭키비키한 기분이야"),
    ("우울/슬픔 계열", "그냥 다 귀찮고 축 처지는 하루야"),
    ("차분/잔잔 계열", "조용히 마음을 가라앉히고 싶어"),
    ("비/장마 계열", "장맛비라 꿀꿀하다"),
    ("맑음/화창 계열", "하늘이 완전 쨍하고 좋다"),
    ("조깅/뛰면서 계열", "지금 헬스장에서 러닝머신 뛰는 중"),
    ("산책/여유 계열", "천천히 동네 한 바퀴 걷는 중"),
    ("관련 없음(매칭 안 되길 기대) 1", "오늘 저녁 뭐 먹을지 고민중이야"),
    ("관련 없음(매칭 안 되길 기대) 2", "내일 회의 몇 시였지?"),
    ("관련 없음(매칭 안 되길 기대) 3", "이 코드 왜 안 돌아가지"),
]


def main():
    groups = _build_concept_groups()
    embedder = _get_embedder()
    from sentence_transformers import util

    print(f"개념(동의어 그룹) 수: {len(groups)}\n")

    all_top1 = {}  # 문장 -> (기대여부, top1 유사도)
    for expected, text in TEST_CASES:
        text_emb = embedder.encode(text, convert_to_tensor=True)
        scored = []
        for keywords, _, examples in groups:
            ex_embs = embedder.encode(examples, convert_to_tensor=True)
            sims = util.cos_sim(text_emb, ex_embs)[0]
            best_idx = int(sims.argmax())
            scored.append((float(sims[best_idx]), keywords[0], examples[best_idx]))
        scored.sort(reverse=True)
        all_top1[text] = scored[0][0]

        print(f"[{expected}] \"{text}\"")
        for sim, kw, ex in scored[:5]:
            print(f"    {kw:6s}  유사도 {sim:.3f}   예문: '{ex}'")
        print()

    irrelevant_max = max(
        sim for (expected, text), sim in zip(TEST_CASES, all_top1.values())
        if "관련 없음" in expected
    )
    relevant_min = min(
        sim for (expected, text), sim in zip(TEST_CASES, all_top1.values())
        if "관련 없음" not in expected
    )
    print(f"'관련 없음' 문장들의 최고 top1 유사도: {irrelevant_max:.3f}")
    print(f"실제 매칭돼야 하는 문장들의 최저 top1 유사도: {relevant_min:.3f}")
    if irrelevant_max < relevant_min:
        suggested = (irrelevant_max + relevant_min) / 2
        print(f"-> 둘 사이 간격이 있음. threshold 추천값: {suggested:.2f}")
    else:
        print(
            "-> 겹치는 구간이 있어서 완벽하게 나눌 수 있는 threshold가 없음.\n"
            "   개별 [카테고리]별 결과를 보고 직접 판단해서 조정하세요."
        )


if __name__ == "__main__":
    main()
