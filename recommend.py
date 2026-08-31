"""
Stage 2 추천 엔진: 사용자가 입력한 곡과 가장 비슷한 곡을 추천.

1단계 K-means는 EDA/시각화 용도로 계속 써도 되지만, "곡 1개 입력 -> 비슷한 곡
추천"이라는 실제 사용 시나리오에는 k-NN(거리 기반)이 더 적합함.
클러스터 경계에 걸쳐있는 곡을 놓치는 문제가 없고, top-N을 자유롭게 조절할 수 있음.

--- 2026-08-21 수정 내역 (ISSUES.md 참고) ---
1) 코사인 유사도 -> 유클리드 거리로 교체.
   StandardScaler로 평균 중심(mean-centered)에 놓인 벡터에 코사인 유사도를 쓰면,
   가장 튀는 특성 하나(이상치)가 유사도 방향을 사실상 독차지해버리는 문제가 있었음.
   (예: bad guy의 speechiness가 z-score +2.63인 이상치라서, speechiness가
   유난히 높은 곡들(뮤지컬/코미디 등)만 추천되던 버그)
2) 장르 우선 필터링 추가.
   오디오 특성만 보고 장르를 전혀 고려하지 않아서, 수치상 우연히 비슷하면
   전혀 다른 장르(동요, 뮤지컬 등)가 섞여 나오는 문제가 있었음. 이제 같은
   장르 후보가 충분히 있으면 그 안에서만 거리로 랭킹을 매기고, 후보가 너무
   적으면(< MIN_GENRE_POOL) 전체 데이터셋으로 자동 확장함.
3) 상황 텍스트 기반 추천(recommend_by_target)에 무드 장르 화이트리스트 추가.
   곡 제목 기반과 달리 "기준 장르"가 없어서(사용자가 곡을 준 게 아니라 텍스트를
   줬으므로) 오디오 수치만으로 찾으면 show-tunes/gospel/cantopop/turkish처럼
   전혀 다른 장르·언어권이 뒤섞여 나왔음. "top-N 후보 중 다수결로 장르 하나
   고르기"도 시도해봤는데, mandopop/cantopop/acoustic이 거의 동률로 나와서
   K값에 따라 결과가 들쭉날쭉해 불안정했음. 그래서 대신 무드 음악으로 흔히
   듣는 장르만 남기는 화이트리스트(MOOD_GENRES) 방식으로 필터링함.
4) 상황 텍스트 기반 추천에 선택적 artist 파라미터 추가.
   artist가 주어지면 그 아티스트가 걸쳐있는 장르(들)로 후보를 좁혀서(곡 기반
   추천의 "같은 장르 우선" 필터링과 같은 개념) 그 안에서 무드 타깃 벡터로
   랭킹을 매김. artist가 없거나 데이터셋에서 못 찾으면 기존 MOOD_GENRES
   화이트리스트 방식으로 자동 대체함.
5) artist가 주어지면 그 아티스트 본인 곡 중 무드에 가장 가까운 곡을 1위로
   고정함(사용자 요청). 예: "칙칙해" + Justin Bieber일 때, 그의 곡들이 원래
   전체 랭킹에서 밀려서(그의 곡 대부분이 밝은/신나는 편이라 "칙칙한" 무드와
   거리가 멀어서) 하나도 안 나왔던 문제가 있었음. 이제 그의 곡 중 그나마
   가장 가까운 곡을 무조건 1위에 넣고, 나머지 자리는 기존처럼(장르 앵커 +
   거리 랭킹) 채움.
"""
import numpy as np
import pandas as pd
from difflib import get_close_matches
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from build_dataset import FEATURE_COLS

# 같은 장르 후보가 이보다 적으면 장르 필터를 포기하고 전체 데이터셋에서 찾음
MIN_GENRE_POOL = 15

# 상황 텍스트 기반(recommend_by_target) 추천에서 후보로 남길 "무드 음악" 장르.
# show-tunes/gospel/kids/disney/comedy/sleep/opera 같은 니치·특수목적 카테고리와
# cantopop/mandopop/turkish/k-pop/j-pop 등 언어권 장르는 기본 제외함 (원하면
# 이 목록에 추가해서 다시 포함시킬 수 있음).
MOOD_GENRES = {
    "acoustic", "alt-rock", "alternative", "ambient", "blues", "chill", "classical",
    "country", "dance", "disco", "drum-and-bass", "dub", "dubstep", "edm", "electro",
    "electronic", "emo", "folk", "funk", "garage", "goth", "groove", "grunge",
    "guitar", "happy", "hard-rock", "heavy-metal", "hip-hop", "house", "indie",
    "indie-pop", "jazz", "metal", "new-age", "piano", "pop", "pop-film",
    "power-pop", "psych-rock", "punk", "punk-rock", "r-n-b", "reggae", "rock",
    "rock-n-roll", "sad", "singer-songwriter", "ska", "soul", "study", "synth-pop",
    "techno", "trance", "trip-hop",
}

# 화이트리스트 적용 후 후보가 이보다 적으면(이론상 거의 없지만 안전장치로) 포기하고
# 전체 데이터셋에서 찾음
MIN_MOOD_POOL = 50


class SongRecommender:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(self.df[FEATURE_COLS])

    def find_song(self, title, artist=None):
        """
        정확히 일치하지 않아도 fuzzy matching으로 가장 비슷한 곡 인덱스 찾기.

        버그 수정(2026-08-21, ISSUES.md 참고): 예전엔 title로 먼저 후보 5개를
        뽑고 그 안에서만 artist를 필터링해서, 정답 곡이 그 5개 안에 없으면
        (예: "lovely" 검색 시 실제 곡명이 "lovely (with Khalid)"라 후보 밖으로
        밀려남) artist를 완전히 무시하고 엉뚱한 동명 곡으로 조용히 대체됐음.
        지금은 artist가 주어지면 "해당 아티스트의 곡들" 안에서 먼저 좁힌 뒤
        title을 fuzzy match해서, 곡 표기가 좀 달라도(feat./with 등) 아티스트가
        맞는 곡을 우선적으로 찾음.
        """
        if artist:
            artist_pool = self.df[
                self.df["artists"].str.contains(artist, case=False, na=False)
            ]
            if not artist_pool.empty:
                # cutoff=0.0: 이미 아티스트로 좁혀놨으니, 표기가 좀 달라도
                # (feat./with 등) 그 아티스트 곡 중 title과 가장 가까운 곡을 반환
                pool_titles = artist_pool["track_name"].tolist()
                pool_matches = get_close_matches(title, pool_titles, n=1, cutoff=0.0)
                return artist_pool[
                    artist_pool["track_name"] == pool_matches[0]
                ].index[0]
            # 이 아티스트로는 아예 못 찾음 -> 아래에서 title만으로 재시도

        titles = self.df["track_name"].tolist()
        matches = get_close_matches(title, titles, n=5, cutoff=0.6)
        if not matches:
            return None

        return self.df[self.df["track_name"] == matches[0]].index[0]

    def recommend(self, title, artist=None, top_n=10, prefer_same_genre=True):
        idx = self.find_song(title, artist)
        if idx is None:
            return None, f"'{title}'을(를) 데이터셋에서 찾을 수 없음 (다른 표기로 시도해볼 것)", None

        query_pos = self.df.index.get_loc(idx)
        note = None

        if prefer_same_genre:
            genre = self.df.loc[idx, "track_genre"]
            genre_mask = (self.df["track_genre"] == genre).to_numpy()
            if genre_mask.sum() >= MIN_GENRE_POOL:
                pool_positions = np.where(genre_mask)[0]
            else:
                pool_positions = np.arange(len(self.df))
                note = (
                    f"'{genre}' 장르 후보가 {MIN_GENRE_POOL}곡 미만이라 "
                    f"전체 데이터셋에서 찾음"
                )
        else:
            pool_positions = np.arange(len(self.df))

        pool_matrix = self.feature_matrix[pool_positions]
        query_vector = self.feature_matrix[query_pos].reshape(1, -1)
        dists = euclidean_distances(query_vector, pool_matrix)[0]

        order = np.argsort(dists)
        order = [i for i in order if pool_positions[i] != query_pos][:top_n]

        result_positions = pool_positions[order]
        result = self.df.iloc[result_positions][["track_name", "artists", "track_genre"]].copy()
        result["distance"] = dists[order]
        return result, None, note

    def recommend_by_target(
        self, target_features, top_n=10, use_mood_whitelist=True, artist=None
    ):
        """
        곡 제목이 아니라 목표 오디오 특징값(dict)을 입력받아 가장 가까운 곡 추천.
        context_parser.parse_context()가 만든 target dict를 그대로 넣으면 됨.
        곡 기반 추천과 달리 "기준이 되는 장르"가 없어서, artist가 주어지면 그
        아티스트의 장르(들)로 먼저 좁히고, 없거나 못 찾으면 MOOD_GENRES
        화이트리스트로 후보를 좁힌 뒤 그 안에서 거리로 랭킹을 매김.
        """
        target_row = pd.DataFrame(
            [[target_features.get(col, 0) for col in FEATURE_COLS]], columns=FEATURE_COLS
        )
        target_scaled = self.scaler.transform(target_row)

        note = None
        pool_positions = None
        pinned_pos = None  # 1위로 고정할 "그 아티스트 본인 곡"의 위치(있으면)

        if artist:
            artist_rows = self.df[
                self.df["artists"].str.contains(artist, case=False, na=False)
            ]
            if artist_rows.empty:
                note = f"'{artist}'을(를) 데이터셋에서 찾을 수 없어서 무드 장르로 대체함"
            else:
                artist_positions = artist_rows.index.to_numpy()
                artist_dists = euclidean_distances(
                    target_scaled, self.feature_matrix[artist_positions]
                )[0]
                pinned_pos = artist_positions[np.argmin(artist_dists)]

                artist_genres = set(artist_rows["track_genre"].unique())
                genre_mask = self.df["track_genre"].isin(artist_genres).to_numpy()
                if genre_mask.sum() >= MIN_GENRE_POOL:
                    pool_positions = np.where(genre_mask)[0]
                    note = (
                        f"1위는 '{artist}' 본인 곡 중 가장 가까운 곡으로 고정, "
                        f"나머지는 '{artist}'의 장르({', '.join(sorted(artist_genres))}) 기준으로 찾음"
                    )
                else:
                    note = (
                        f"1위는 '{artist}' 본인 곡 중 가장 가까운 곡으로 고정, "
                        f"나머지는 장르 후보가 너무 적어서 무드 장르로 대체함"
                    )

        if pool_positions is None:
            if use_mood_whitelist:
                mood_mask = self.df["track_genre"].isin(MOOD_GENRES).to_numpy()
                if mood_mask.sum() >= MIN_MOOD_POOL:
                    pool_positions = np.where(mood_mask)[0]
                else:
                    pool_positions = np.arange(len(self.df))
                    note = (note + " / " if note else "") + (
                        "무드 장르 화이트리스트 후보가 너무 적어서 전체 데이터셋에서 찾음"
                    )
            else:
                pool_positions = np.arange(len(self.df))

        pool_matrix = self.feature_matrix[pool_positions]
        dists = euclidean_distances(target_scaled, pool_matrix)[0]

        order = np.argsort(dists)
        if pinned_pos is not None:
            # 1위로 고정한 곡은 "나머지" 랭킹에서 중복으로 또 뽑히지 않게 제외
            order = [i for i in order if pool_positions[i] != pinned_pos]
        order = order[: (top_n - 1 if pinned_pos is not None else top_n)]

        result_positions = list(pool_positions[order])
        result_dists = list(dists[order])

        if pinned_pos is not None:
            pinned_distance = euclidean_distances(
                target_scaled, self.feature_matrix[pinned_pos].reshape(1, -1)
            )[0][0]
            result_positions = [pinned_pos] + result_positions
            result_dists = [pinned_distance] + result_dists

        result = self.df.iloc[result_positions][["track_name", "artists", "track_genre"]].copy()
        result["distance"] = result_dists
        return result, note


if __name__ == "__main__":
    rec = SongRecommender("data/stage2_dataset.csv")
    result, error, note = rec.recommend("bad guy", artist="Billie Eilish", top_n=10)
    if error:
        print(error)
    else:
        if note:
            print(note)
        print(result.to_string(index=False))