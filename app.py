"""
Stage 3/4: Stage 2 추천 엔진을 웹 UI로 감싼 Streamlit 앱.

recommend.py / context_parser.py / build_dataset.py는 그대로 재사용하고,
이 파일은 UI 레이어만 담당함. 무거운 데이터 로딩(SongRecommender 초기화 -
33,992곡 CSV 읽기 + StandardScaler.fit_transform)은 매 상호작용마다 반복하면
느려지므로 st.cache_resource로 앱 시작 시 한 번만 로드해서 재사용함.

--- 2026-08-22 수정: 좋아요/스킵 피드백 수집 (stage4 준비) ---
stage4에서 재랭킹 모델(지도학습)을 학습시키려면 "이 사용자가 이 곡을
좋아했는지"에 대한 라벨 데이터가 필요함. 그 데이터를 모으는 첫 단계로 결과
카드마다 좋아요/스킵 버튼을 붙이고, 클릭할 때마다 FEEDBACK_PATH(feedback.csv)에
한 줄씩 기록함. data/*.csv는 .gitignore에 걸려있어서(원본 Kaggle 데이터용
규칙) feedback.csv는 일부러 data/ 밖(프로젝트 루트)에 둠 - 이건 우리가 직접
만든 학습 데이터라 git으로 계속 추적하고 싶어서.

버튼을 누르면 Streamlit이 스크립트를 처음부터 다시 실행하는데, 그때 추천
결과를 다시 계산하지 않으면(버튼이 안에 있던 if 블록이 이번 실행에선 False라서)
카드 자체가 사라져버림. 그래서 추천 결과를 st.session_state에 저장해두고,
"추천받기" 버튼을 누른 그 실행이든 아니든 항상 session_state에서 읽어서
렌더링하도록 구조를 바꿈.

--- 2026-08-22 수정: 가벼운 사용자 구분(user_id) 추가 ---
피드백이 전부 한 파일(feedback.csv)에 섞이면, 나중에 이 앱을 배포해서 다른
사람(예: 리크루터)이 몇 번 눌러본 클릭까지 "내 취향 데이터"에 섞여 들어가서
재랭킹 모델이 개인화가 아니라 여러 사람 취향이 뒤섞인 모델이 돼버림. 정식
로그인(계정/비밀번호/DB)은 이 프로젝트의 핵심(추천 ML)과 무관한 인프라라
과함. 대신 닉네임만 입력받아 feedback.csv에 user_id로 남기는 가벼운 방식을
씀 - 인증은 없지만(비밀번호 없음, 누구나 아무 이름이나 칠 수 있음) 적어도
"누구 피드백인지" 구분은 되고, 나중에 재랭킹 모델을 "특정 user_id의 피드백만"
필터링해서 학습시킬 수 있음.
"""
import csv
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from recommend import SongRecommender
from context_parser import parse_context

DATASET_PATH = "data/stage2_dataset.csv"
FEEDBACK_PATH = "feedback.csv"
FEEDBACK_FIELDS = [
    "timestamp", "user_id", "source", "context", "track_name", "artists",
    "track_genre", "distance", "label",
]
DEFAULT_USER_ID = "guest"

# --- 2026-08-31 추가: 한국어/영어 UI 토글 ---
# 리크루터에게 보여줄 때는 영어로 보여줘야 해서, UI 문구(라벨/버튼/캡션)를
# 전부 영어로 바꿀 수 있는 토글을 추가함. 다만 상황 텍스트 매칭 로직
# (KEYWORD_RULES, CONCEPT_EXAMPLE_SENTENCES)은 한국어 문장에 맞춰
# 만들어진 거라서, 실제 "지금 기분" 입력창은 영어 모드에서도 한국어로
# 입력해야 의도대로 동작함 - 이건 번역이 아니라 한국어 NLP를 보여주는
# 프로젝트라서, 영어 모드에서는 대신 안내 문구를 하나 더 보여줌.
LANG_LABELS = {"ko": "한국어", "en": "English"}

T = {
    "page_title": {"ko": "무드 노래 추천", "en": "Mood Song Recommender"},
    "app_title": {"ko": "🎧 무드 노래 추천", "en": "🎧 Mood Song Recommender"},
    "app_caption": {
        "ko": "stage2 추천 엔진(장르 우선 필터링 + 유클리드 거리)을 그대로 쓰는 웹 버전. "
              "곡 제목으로 비슷한 곡을 찾거나, 지금 기분을 문장으로 적어서 추천받을 수 있어요.",
        "en": "A web version of the stage2 recommendation engine (genre-first filtering + "
              "Euclidean distance). Find songs similar to one you like, or describe your "
              "mood in a sentence to get recommendations.",
    },
    "sidebar_user_header": {"ko": "👤 사용자", "en": "👤 User"},
    "sidebar_user_caption": {
        "ko": "비밀번호 없는 가벼운 구분용이에요. 좋아요/스킵을 이 이름으로 기록해요.",
        "en": "A lightweight, password-free identifier. Your likes/skips are logged under this name.",
    },
    "nickname_label": {"ko": "닉네임", "en": "Nickname"},
    "nickname_placeholder": {"ko": "예: 민", "en": "e.g. Min"},
    "current_user": {"ko": "현재: **{user_id}**", "en": "Current: **{user_id}**"},
    "feedback_counts": {
        "ko": "{user_id}: {mine}개  \n전체 피드백: {total}개 (stage4 재랭킹 모델 학습용)",
        "en": "{user_id}: {mine}  \nTotal feedback: {total} (training data for the stage4 re-ranking model)",
    },
    "new_feedback_alert": {
        "ko": "🔔 배포 이후 새 피드백 {n}개 - 재부팅 전에 다운로드하세요",
        "en": "🔔 {n} new feedback entries since deploy - download before this app reboots",
    },
    "no_new_feedback": {
        "ko": "현재 배포 상태와 동일 (새로 쌓인 피드백 없음)",
        "en": "Matches the current deployment (no new feedback)",
    },
    "download_button": {"ko": "⬇️ feedback.csv 다운로드", "en": "⬇️ Download feedback.csv"},
    "download_help": {
        "ko": "지금 서버에 쌓인 전체 피드백을 내려받아요. "
              "이 앱이 재부팅되면 여기서 받은 것 외의 기록은 사라져요.",
        "en": "Downloads all feedback currently on the server. Anything not downloaded "
              "here is lost when this app reboots.",
    },
    "tab_song": {"ko": "🎵 곡으로 찾기", "en": "🎵 Find by song"},
    "tab_mood": {"ko": "💭 지금 기분으로 찾기", "en": "💭 Find by mood"},
    "song_subheader": {"ko": "좋아하는 곡과 비슷한 곡 찾기", "en": "Find songs similar to one you like"},
    "song_query_label": {"ko": "곡 제목을 입력해보세요", "en": "Enter a song title"},
    "song_query_placeholder": {"ko": "예: bad guy", "en": "e.g. bad guy"},
    "song_not_found": {
        "ko": "'{query}'와 비슷한 곡을 찾을 수 없어요. 다른 표기로 시도해보세요.",
        "en": "Couldn't find anything close to '{query}'. Try a different spelling.",
    },
    "song_choice_label": {"ko": "찾는 곡이 있으면 골라주세요", "en": "Pick the song you meant, if it's here"},
    "song_recommend_button": {"ko": "이 곡과 비슷한 곡 추천받기", "en": "Recommend songs like this"},
    "mood_subheader": {
        "ko": "지금 기분/날씨/걷는 속도로 찾기",
        "en": "Find by mood / weather / walking pace",
    },
    "mood_text_label": {
        "ko": "지금 상황을 자유롭게 적어보세요",
        "en": "Describe your current mood freely",
    },
    "mood_text_placeholder": {"ko": "예: 오늘 비오고 기분 칙칙해", "en": "예: 오늘 비오고 기분 칙칙해"},
    "mood_ko_only_hint": {
        "ko": "",
        "en": "ℹ️ This project's mood/weather matching (rule-based keywords + Korean sentence "
              "embeddings) is built for Korean text. Type your mood in Korean (like the "
              "placeholder above) to see it match as intended.",
    },
    "mood_artist_label": {"ko": "선호하는 가수 (필수)", "en": "Preferred artist (required)"},
    "mood_artist_placeholder": {"ko": "예: Billie Eilish", "en": "e.g. Billie Eilish"},
    "mood_artist_required_error": {
        "ko": "선호하는 가수를 입력해주세요 - 없으면 추천 결과가 너무 다양한 장르로 흩어져요.",
        "en": "Please enter a preferred artist - without one, results can end up scattered across too many unrelated genres.",
    },
    "mood_recommend_button": {"ko": "지금 상황에 어울리는 곡 추천받기", "en": "Recommend songs for this mood"},
    "no_keywords_matched": {
        "ko": "인식된 키워드가 없어서 데이터셋 평균값 기준으로 추천했어요.",
        "en": "No keywords were recognized, so recommendations are based on the dataset average.",
    },
    "matched_keywords": {"ko": "인식된 키워드: {keywords}", "en": "Recognized keywords: {keywords}"},
    "like_button": {"ko": "👍 좋아요", "en": "👍 Like"},
    "skip_button": {"ko": "👎 스킵", "en": "👎 Skip"},
    "liked_label": {"ko": "🙏 좋아요 기록됨", "en": "🙏 Liked - recorded"},
    "skipped_label": {"ko": "🙏 스킵 기록됨", "en": "🙏 Skipped - recorded"},
    "change_vote_button": {"ko": "변경", "en": "Change"},
    "distance_caption": {"ko": "거리 {d:.3f} (작을수록 유사)", "en": "distance {d:.3f} (lower = more similar)"},
}


def t(key, **kwargs):
    lang = st.session_state.get("lang", "ko")
    text = T[key][lang]
    return text.format(**kwargs) if kwargs else text


st.session_state.setdefault("lang", "ko")
st.set_page_config(page_title=t("page_title"), page_icon="🎧", layout="centered")


@st.cache_resource(show_spinner="데이터셋 불러오는 중...")
def load_recommender():
    return SongRecommender(DATASET_PATH)


@st.cache_data(show_spinner=False)
def search_titles(query, top_n=15):
    """
    검색창에 입력한 문자열로 후보 곡을 찾는다. 먼저 제목에 부분 문자열로
    포함되는 곡을 우선 찾고(빠르고 직관적), 없으면 오타를 감안해 fuzzy
    matching으로 재시도한다.
    """
    rec = load_recommender()
    df = rec.df
    if not query:
        return df.iloc[0:0]

    mask = df["track_name"].str.contains(query, case=False, na=False)
    candidates = df[mask]

    if candidates.empty:
        from difflib import get_close_matches

        matches = get_close_matches(
            query, df["track_name"].tolist(), n=top_n, cutoff=0.5
        )
        candidates = df[df["track_name"].isin(matches)]

    # popularity 높은 순으로 정렬해서 그나마 더 알려진 곡을 위로
    return candidates.sort_values("popularity", ascending=False).head(top_n)


def append_feedback(user_id, source, context, row, label):
    """
    (user_id, source, context, track_name, artists) 조합이 이미 있으면 그
    이전 기록을 지우고 새로 쓴다(최신 투표로 덮어씀) - 그냥 계속 append만
    하면 같은 카드에 여러 번 투표할 때(버튼이 안 없어져서 여러 번 눌리는
    경우, 페이지 새로고침 후 다시 누르는 경우 등) feedback.csv에 중복
    행이 쌓여서 "내 피드백 개수"가 실제 좋아요/스킵한 곡 수보다 부풀려지는
    버그가 있었음(2026-08-31, ISSUES.md 참고). 세션 쪽에서도 이미 투표한
    카드는 버튼을 숨기지만, 새로고침/재실행에도 안전하도록 저장 단에서도
    한 번 더 막아둠.
    """
    new_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "source": source,
        "context": context,
        "track_name": row["track_name"],
        "artists": row["artists"],
        "track_genre": row["track_genre"],
        "distance": row["distance"],
        "label": label,
    }

    existing_rows = []
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    key = (user_id, source, context, row["track_name"], row["artists"])
    existing_rows = [
        r for r in existing_rows
        if (r["user_id"], r["source"], r["context"], r["track_name"], r["artists"]) != key
    ]
    existing_rows.append(new_row)

    with open(FEEDBACK_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)


def count_feedback(user_id=None):
    """user_id를 주면 그 사람 것만, 안 주면 전체 행 수를 센다."""
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    df = pd.read_csv(FEEDBACK_PATH)
    if user_id:
        df = df[df["user_id"] == user_id]
    return len(df)


_baseline_feedback_count = None


def get_baseline_feedback_count():
    """
    이번 배포(재부팅) 시점에 feedback.csv에 있던 행 수를 딱 한 번만 계산해서
    고정해둠. 처음엔 st.cache_resource로 구현했는데, 그건 앱 화면의
    "Clear cache" 메뉴/단축키(C)로 누구나 지울 수 있는 캐시라서 - 방문자가
    실수로 그걸 누르면 기준값이 그 순간으로 재설정되면서 실제로는 새
    피드백이 있는데도 "없음"으로 잘못 표시되는 문제가 있었음. 그래서
    Streamlit 캐시 시스템 밖에 있는 일반 모듈 전역 변수로 바꿈 - 이건
    Clear cache로는 안 지워지고, 앱이 실제로 재부팅(프로세스 재시작)될
    때만 초기화됨.
    """
    global _baseline_feedback_count
    if _baseline_feedback_count is None:
        _baseline_feedback_count = count_feedback()
    return _baseline_feedback_count


def render_result_cards(results, source, context, user_id):
    for i, row in results.iterrows():
        vote_key = f"vote_{source}_{i}_{row['track_name']}_{row['artists']}"
        with st.container(border=True):
            st.markdown(f"**{row['track_name']}**")
            st.caption(f"{row['artists']} · {row['track_genre']}")
            st.progress(
                max(0.0, min(1.0, 1 - row["distance"] / 3)),
                text=t("distance_caption", d=row["distance"]),
            )

            current_vote = st.session_state.get(vote_key)
            if current_vote is None:
                # 아직 투표 안 한 카드만 버튼을 보여줌 - 투표 후에도 버튼이
                # 계속 남아있으면 다시 눌러서 feedback.csv에 중복 행이
                # 쌓이는 문제가 있었음(2026-08-31 수정).
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button(t("like_button"), key=f"{vote_key}_like"):
                        append_feedback(user_id, source, context, row, "like")
                        st.session_state[vote_key] = "like"
                        st.rerun()
                with col2:
                    if st.button(t("skip_button"), key=f"{vote_key}_skip"):
                        append_feedback(user_id, source, context, row, "skip")
                        st.session_state[vote_key] = "skip"
                        st.rerun()
            else:
                label = t("liked_label") if current_vote == "like" else t("skipped_label")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(label)
                with col2:
                    if st.button(t("change_vote_button"), key=f"{vote_key}_reset"):
                        st.session_state[vote_key] = None
                        st.rerun()


with st.sidebar:
    lang_choice = st.radio(
        "Language / 언어",
        options=["ko", "en"],
        format_func=lambda code: LANG_LABELS[code],
        index=["ko", "en"].index(st.session_state["lang"]),
        horizontal=True,
        key="lang_radio",
    )
    st.session_state["lang"] = lang_choice
    st.divider()

st.title(t("app_title"))
st.caption(t("app_caption"))

with st.sidebar:
    st.subheader(t("sidebar_user_header"))
    st.caption(t("sidebar_user_caption"))
    nickname = st.text_input(
        t("nickname_label"),
        value=st.session_state.get("nickname", ""),
        placeholder=t("nickname_placeholder"),
    )
    st.session_state["nickname"] = nickname
    user_id = nickname.strip() or DEFAULT_USER_ID
    st.caption(t("current_user", user_id=user_id))
    st.divider()
    st.caption(
        t("feedback_counts", user_id=user_id, mine=count_feedback(user_id), total=count_feedback())
    )

    # --- 2026-08-31 추가: 서버에 쌓인 feedback.csv 다운로드 ---
    # Streamlit Cloud의 로컬 파일시스템은 임시(ephemeral)라 앱이 재부팅되면
    # (12시간 무접속 sleep, 재배포, 리소스 초과 재시작 등) feedback.csv에
    # append된 내용이 사라짐. git에 커밋된 스냅샷만 다음 배포에 남음.
    # 배포된 앱에서 실제로 쌓인 피드백을 영구 보관하려면, 사라지기 전에
    # 이 버튼으로 다운로드해서 로컬 feedback.csv와 합친 뒤 git commit/push
    # 해야 함. (서버에 SSH/파일탐색기로 직접 접근할 방법이 없어서 앱 자체에
    # 다운로드 기능을 넣어야 함.)
    new_since_deploy = count_feedback() - get_baseline_feedback_count()
    if new_since_deploy > 0:
        st.warning(t("new_feedback_alert", n=new_since_deploy))
    else:
        st.caption(t("no_new_feedback"))

    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "rb") as fh:
            st.download_button(
                t("download_button"),
                data=fh.read(),
                file_name="feedback.csv",
                mime="text/csv",
                help=t("download_help"),
            )

tab_song, tab_mood = st.tabs([t("tab_song"), t("tab_mood")])

with tab_song:
    st.subheader(t("song_subheader"))
    query = st.text_input(t("song_query_label"), placeholder=t("song_query_placeholder"), key="song_query")

    if query:
        candidates = search_titles(query)
        if candidates.empty:
            st.warning(t("song_not_found", query=query))
        else:
            options = [
                f"{r.track_name} — {r.artists} ({r.track_genre})"
                for r in candidates.itertuples()
            ]
            choice = st.selectbox(t("song_choice_label"), options, key="song_choice")
            chosen_row = candidates.iloc[options.index(choice)]

            if st.button(t("song_recommend_button"), type="primary"):
                rec = load_recommender()
                results, error, note = rec.recommend(
                    chosen_row["track_name"], chosen_row["artists"], top_n=10,
                    lang=st.session_state["lang"],
                )
                st.session_state["song_results"] = results
                st.session_state["song_error"] = error
                st.session_state["song_note"] = note
                st.session_state["song_context"] = (
                    f"song:{chosen_row['track_name']}|artist:{chosen_row['artists']}"
                )

    if st.session_state.get("song_error"):
        st.error(st.session_state["song_error"])
    elif st.session_state.get("song_results") is not None:
        if st.session_state.get("song_note"):
            st.info(st.session_state["song_note"])
        render_result_cards(
            st.session_state["song_results"], "song", st.session_state["song_context"], user_id
        )

with tab_mood:
    st.subheader(t("mood_subheader"))
    text = st.text_input(
        t("mood_text_label"),
        placeholder=t("mood_text_placeholder"),
        key="mood_text",
    )
    if st.session_state["lang"] == "en":
        st.caption(t("mood_ko_only_hint"))
    artist = st.text_input(
        t("mood_artist_label"), placeholder=t("mood_artist_placeholder"), key="mood_artist"
    )

    if st.button(t("mood_recommend_button"), type="primary"):
        if not artist.strip():
            # 2026-08-31 추가: 아티스트 없이 무드만으로 추천하면 MOOD_GENRES
            # 화이트리스트(약 50개 장르) 전체에서 찾다 보니 장르가 너무
            # 넓게 흩어져서 "이상한 곡"이 자주 나온다는 피드백이 있었음.
            # recommend_by_target()이 artist가 주어지면 그 아티스트의
            # 장르로 후보를 먼저 좁히는 로직이 이미 있어서, 아티스트를
            # 필수 입력으로 강제하면 그 좁히기 로직이 항상 작동함.
            st.warning(t("mood_artist_required_error"))
        else:
            rec = load_recommender()
            target, matched = parse_context(text, rec.df, lang=st.session_state["lang"])
            results, note = rec.recommend_by_target(
                target, top_n=10, artist=artist, lang=st.session_state["lang"]
            )

            st.session_state["mood_results"] = results
            st.session_state["mood_note"] = note
            st.session_state["mood_matched"] = matched
            st.session_state["mood_context"] = f"mood:{text}|artist:{artist}"

    if st.session_state.get("mood_results") is not None:
        if not st.session_state.get("mood_matched"):
            st.info(t("no_keywords_matched"))
        else:
            st.write(t("matched_keywords", keywords=", ".join(st.session_state["mood_matched"])))
        if st.session_state.get("mood_note"):
            st.info(st.session_state["mood_note"])
        render_result_cards(
            st.session_state["mood_results"], "mood", st.session_state["mood_context"], user_id
        )
