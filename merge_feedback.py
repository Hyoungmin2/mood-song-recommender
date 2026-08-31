"""
배포된 앱(Streamlit Cloud)에서 다운로드한 feedback.csv를 로컬 feedback.csv와
합치는 스크립트.

Streamlit Cloud의 로컬 파일시스템은 임시(ephemeral)라 앱이 재부팅되면
(12시간 무접속 sleep, git push로 인한 재배포, 리소스 초과 재시작 등)
서버에 쌓인 피드백이 사라짐(ISSUES.md의 "Streamlit Cloud 배포" 섹션 참고).
그래서 배포된 앱 사이드바의 다운로드 버튼으로 미리 받아둔 파일을, 사라지기
전에 이 스크립트로 로컬 파일과 합쳐서 git에 커밋해야 영구 보관됨.

사용법:
    python3 merge_feedback.py <다운로드한 feedback.csv 경로>

    예: python3 merge_feedback.py ~/Downloads/feedback.csv

동작:
    (user_id, source, context, track_name, artists) 조합을 기준으로
    로컬 feedback.csv와 다운로드한 파일을 병합함. app.py의
    append_feedback()과 동일한 키 기준이라, 겹치는 조합이 있으면
    다운로드한(서버) 쪽 기록으로 덮어씀(더 최근에 받은 값 우선).
    결과는 timestamp 순으로 정렬해서 다시 feedback.csv에 저장함.
"""
import csv
import sys

FIELDS = [
    "timestamp", "user_id", "source", "context", "track_name", "artists",
    "track_genre", "distance", "label",
]
LOCAL_PATH = "feedback.csv"


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(row):
    return (row["user_id"], row["source"], row["context"], row["track_name"], row["artists"])


def main():
    if len(sys.argv) != 2:
        print("사용법: python3 merge_feedback.py <다운로드한 feedback.csv 경로>")
        sys.exit(1)

    downloaded_path = sys.argv[1]

    local_rows = load(LOCAL_PATH)
    downloaded_rows = load(downloaded_path)

    merged = {key(r): r for r in local_rows}
    added = 0
    updated = 0
    for r in downloaded_rows:
        k = key(r)
        if k not in merged:
            added += 1
        elif merged[k]["timestamp"] != r["timestamp"]:
            updated += 1
        merged[k] = r

    rows_sorted = sorted(merged.values(), key=lambda r: r["timestamp"])

    with open(LOCAL_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(
        f"병합 완료: 로컬 {len(local_rows)}개 + 다운로드 {len(downloaded_rows)}개 "
        f"-> 총 {len(rows_sorted)}개 (새로 추가: {added}개, 갱신: {updated}개)"
    )


if __name__ == "__main__":
    main()
