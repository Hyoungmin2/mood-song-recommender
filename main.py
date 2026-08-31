"""
Stage 2 CLI

두 가지 입력 모드를 지원:
1) 곡 제목 기반: 좋아하는 곡과 비슷한 곡 추천 (콘텐츠 기반, 장르 우선 필터링 +
   유클리드 거리)
2) 상황 텍스트 기반: "오늘 비오고 기분 칙칙해" 같은 문장을 입력하면
   기분/날씨/걷는 속도 키워드를 오디오 특징 목표값으로 변환해서 그에 맞는 곡 추천
   (context_parser.py의 규칙 기반 매핑 + 같은 거리 기반 엔진 재사용)

참고: 원래는 Spotify Web API(spotify_client.py)로 곡 존재 여부 확인과
인기도/링크 같은 메타데이터를 보강할 계획이었으나, 2026년 2월 Spotify의
Developer Mode 정책 변경(Premium 계정 필수화)으로 인해 API 의존 없이
Kaggle 데이터셋만으로 동작하도록 단순화함.
"""
import pandas as pd
from recommend import SongRecommender
from context_parser import parse_context


def print_results(results, title):
    print(f"\n{title}\n")
    for _, row in results.iterrows():
        print(
            f"- {row['track_name']} / {row['artists']} ({row['track_genre']}) "
            f"| 거리 {row['distance']:.3f} (작을수록 유사)"
        )


def main():
    rec = SongRecommender("data/stage2_dataset.csv")

    print("어떤 방식으로 추천받을래?")
    print("1) 좋아하는 곡 제목으로 비슷한 곡 찾기")
    print("2) 지금 기분/날씨/걷는 속도를 문장으로 입력하기")
    mode = input("선택 (1 또는 2): ").strip()

    if mode == "1":
        title = input("좋아하는 곡 제목: ").strip()
        artist = input("아티스트 (모르면 그냥 엔터): ").strip() or None

        results, error, note = rec.recommend(title, artist, top_n=10)
        if error:
            print(error)
            return
        if note:
            print(f"\n⚠ {note}")
        print_results(results, f"'{title}'와 비슷한 곡 추천:")

    elif mode == "2":
        text = input("지금 상황을 자유롭게 적어줘 (예: 오늘 비오고 기분 칙칙해): ").strip()
        artist = input("선호하는 가수 (없으면 그냥 엔터): ").strip() or None

        target, matched = parse_context(text, rec.df)
        if not matched:
            print("\n⚠ 인식된 키워드가 없어서 데이터셋 평균값 기준으로 추천함")
        else:
            print(f"\n인식된 키워드: {', '.join(matched)}")

        results, note = rec.recommend_by_target(target, top_n=10, artist=artist)
        if note:
            print(f"⚠ {note}")
        print_results(results, "지금 상황에 어울리는 곡 추천:")

    else:
        print("1 또는 2를 입력해줘")


if __name__ == "__main__":
    main()