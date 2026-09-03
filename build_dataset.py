"""
Kaggle 'Spotify Tracks Dataset' 확장 + 장르 균등 샘플링

다운로드: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
(약 11만 4천 곡, danceability/energy/valence/tempo 등 audio feature가 이미 포함됨)

기존에는 장르별로 최대 150곡씩만 샘플링했는데, 확인해보니 인기도 40 이상 기준으로
500곡 넘게 있는 장르가 26개나 돼서 큰 장르는 실제 후보의 최대 70%가 랜덤으로
버려지고 있었음. recommend.py가 이제 "같은 장르 우선 필터링"을 하기 때문에
장르별 후보 풀이 클수록 유리해짐 -> 기본값을 캡 없음(전체 사용)으로 변경.
popularity>=40 필터만 거친 전체 데이터가 3만4천 곡 수준이라 pandas/sklearn으로
처리하는 데 전혀 부담 없음.
"""
import re

import pandas as pd

FEATURE_COLS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "liveness",
    "speechiness", "loudness",
]


def normalize_title(title):
    """
    "Moral of the Story"/"Moral of the Story (feat. Niall Horan)"/
    "Moral of the Story (feat. Niall Horan) - Bonus Track"처럼, 같은 곡의
    다른 에디션(피처링 표기, 리마스터, 라이브 버전, 스페드업 등)이 Kaggle
    원본에 각각 다른 track_id로 따로 들어있어서 추천 결과에 사실상 같은 곡이
    여러 번 나오는 문제가 있었음(2026-09-03, ISSUES.md 참고). 괄호 안 내용과
    흔한 에디션 접미사를 제거해서 "같은 곡"으로 묶기 위한 정규화 함수.
    """
    t = title.lower()
    t = re.sub(r"\(.*?\)", "", t)  # 괄호 안 내용 제거: (feat. ...), (Live), (Remastered) 등
    t = re.sub(
        r"\s*-\s*(bonus track|deluxe( edition)?|remaster(ed)?( \d{4})?|"
        r"radio edit|extended( mix)?|single version|album version|"
        r"sped up|slowed( \+ reverb)?|8d version|"
        r"acoustic( version)?|live( version)?|explicit|clean).*$",
        "", t,
    )
    return t.strip()


def drop_duplicate_editions(df):
    """
    같은 아티스트 + 정규화한 제목이 같은 행이 여러 개면, popularity가 가장 높은
    버전 하나만 남김. exact-match 중복(drop_duplicates)으로는 못 잡는, "사실상
    같은 곡인데 제목 표기만 다른" 케이스를 잡기 위한 것.
    """
    before = len(df)
    df = df.copy()
    df["_norm_title"] = df["track_name"].apply(normalize_title)
    df = (
        df.sort_values("popularity", ascending=False)
        .drop_duplicates(subset=["artists", "_norm_title"], keep="first")
        .drop(columns=["_norm_title"])
    )
    print(f"같은 곡의 다른 에디션 중복 제거: {before}곡 -> {len(df)}곡")
    return df


def load_raw_dataset(csv_path, min_popularity=40):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURE_COLS + ["track_name", "artists", "track_genre"])
    df = df.drop_duplicates(subset=["track_name", "artists"])
    df = drop_duplicate_editions(df)
    # 인기도가 너무 낮은 곡은 제외. 이 데이터셋이 2022년경 스냅샷이라 release_date가
    # 없어서 "최신곡"을 직접 거를 수는 없지만, popularity가 어느 정도 있는 곡들은
    # 최근까지도 꾸준히 스트리밍되는 곡일 가능성이 높아서 체감 품질이 나아짐.
    before = len(df)
    df = df[df["popularity"] >= min_popularity]
    print(f"인기도 {min_popularity} 미만 제외: {before}곡 -> {len(df)}곡")
    return df


def stratified_sample(df, n_per_genre=None, random_state=42):
    """
    장르별로 최대 n_per_genre곡씩 뽑아 장르 분포를 균등하게 맞춤.
    n_per_genre=None이면 샘플링하지 않고 전체를 그대로 사용함 (장르별 곡수는
    쏠리지만, recommend.py가 장르 우선 필터링을 하므로 후보 풀이 큰 게 더 유리함).
    """
    if n_per_genre is None:
        return df.reset_index(drop=True)

    sampled = (
        df.groupby("track_genre", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), n_per_genre), random_state=random_state))
    )
    return sampled.reset_index(drop=True)


if __name__ == "__main__":
    RAW_PATH = "data/spotify_tracks_dataset.csv"  # Kaggle에서 받은 원본 CSV 경로로 수정
    OUT_PATH = "data/stage2_dataset.csv"

    df = load_raw_dataset(RAW_PATH)
    print(f"원본: {len(df)}곡 / 장르 수: {df['track_genre'].nunique()}")

    sample = stratified_sample(df, n_per_genre=None)  # 캡 없이 전체 사용
    print(f"저장 대상: {len(sample)}곡 / 장르 수: {sample['track_genre'].nunique()}")

    sample.to_csv(OUT_PATH, index=False)
    print(f"저장 완료: {OUT_PATH}")