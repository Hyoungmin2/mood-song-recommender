# 무드 노래 추천 (Mood Song Recommender)

기분/날씨/걷는 속도 같은 자연어 문장이나, 좋아하는 곡 제목으로 비슷한 노래를
추천해주는 개인 프로젝트. Kaggle의 [Spotify Tracks
Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
(오디오 특징 + 장르 라벨이 포함된 곡 약 11만 개)을 기반으로 한다.

Streamlit으로 만든 웹 앱(`app.py`)이 최종 결과물이고, 그 아래에는 콘텐츠
기반 추천 엔진, 자연어 상황 파서, 좋아요/스킵 피드백 수집 기능이 있다.

## 주요 기능

- **곡 기반 추천**: 곡 제목을 입력하면 오디오 특징(danceability, energy,
  valence, tempo 등)과 장르를 기준으로 가장 비슷한 곡 10개를 추천.
- **상황 텍스트 기반 추천**: "비 오고 차분함", "신나고 조깅하는 중" 같은
  자유 문장을 입력하면, 그 문장에서 기분/날씨/걷는 속도를 읽어서 어울리는
  곡을 추천. 원하는 가수를 같이 입력하면 그 가수의 장르 안에서 찾고, 그
  가수 본인 곡 중 가장 어울리는 곡을 1위로 고정해서 보여준다.
- **의미 기반 문장 이해**: 위 상황 텍스트 파서는 미리 정해둔 키워드
  사전(규칙 기반)뿐 아니라, 한국어 문장 임베딩 모델(`jhgan/ko-sroberta-multitask`)로
  사전에 없는 표현("장맛비라 꿀꿀하다" 등)도 의미가 비슷한 키워드로 매칭한다
  (선택 기능 - 아래 설치 방법 참고).
- **피드백 수집**: 추천 결과마다 좋아요/스킵을 남길 수 있고, 닉네임으로
  사용자를 가볍게 구분해서 기록한다. 이 데이터는 다음 단계(피드백 기반
  재랭킹 모델)의 학습 데이터로 쓸 계획.

## 기술 스택

Python, pandas, scikit-learn(StandardScaler, 유클리드 거리 기반 k-NN),
Streamlit, sentence-transformers(선택 기능).

## 프로젝트 구조

```
stage2/
├── app.py                          # Streamlit 웹 앱 (최종 UI)
├── main.py                         # CLI 버전 (앱과 같은 추천 로직 확인용)
├── build_dataset.py                # Kaggle 원본 CSV -> 정제된 데이터셋
├── recommend.py                    # 추천 엔진 (곡 기반 / 상황 기반)
├── context_parser.py               # 자연어 문장 -> 오디오 특징 목표값 변환
├── calibrate_embedding_threshold.py# 임베딩 매칭 threshold 검증용 스크립트
├── spotify_client.py               # (선택) Spotify Web API 검색 연동
├── requirements.txt                # 핵심 의존성
├── requirements-embedding.txt      # 임베딩 매칭용 선택 의존성 (무거움)
└── ISSUES.md                       # 개발하면서 겪은 문제/원인/해결 기록
```

## 설치 및 실행

### 1. 데이터셋 준비

Kaggle에서 [Spotify Tracks
Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)을
받아서 `data/spotify_tracks_dataset.csv`로 저장한다 (용량 문제로 이 저장소에는
원본 데이터를 포함하지 않음).

```
python3 build_dataset.py
```

를 실행하면 `data/stage2_dataset.csv`(정제된 데이터셋)가 생성된다.

### 2. 핵심 의존성 설치

```
pip3 install -r requirements.txt
```

### 3. (선택) 임베딩 기반 의미 매칭 활성화

```
pip3 install -r requirements-embedding.txt
python3 calibrate_embedding_threshold.py   # threshold 검증(선택)
```

설치 안 해도 앱은 정상 동작한다 (규칙 기반 키워드 매칭만 사용, 자동 폴백).

### 4. 실행

```
streamlit run app.py
```

CLI 버전으로 로직만 빠르게 확인하고 싶으면 `python3 main.py`.

## 개발 히스토리

이 프로젝트는 4단계로 진행했다.

1. **stage1**: K-means로 곡 10개를 클러스터링해보는 최소 동작 버전.
2. **stage2**: 실제 Kaggle 데이터셋으로 콘텐츠 기반 추천 구현. 코사인
   유사도가 이상치 특성에 휘둘리는 문제, 장르 필터링 부재로 결과가 뒤섞이는
   문제 등을 발견하고 수정.
3. **stage3**: Streamlit으로 웹 UI 제작, 좋아요/스킵 피드백 수집 기능 추가.
4. **stage4**: ML 요소 강화 - 상황 텍스트 파서에 임베딩 기반 의미 매칭 추가
   (진행), 피드백 기반 재랭킹 모델(예정), stage1 K-means를 취향 클러스터링
   으로 부활(예정), 정량 평가(예정).

문제를 발견하고, 원인을 확인하고, 해결책을 검토하고, 재검증한 전체 과정은
[`ISSUES.md`](./ISSUES.md)에 시간순으로 기록해뒀다.

## 향후 계획

- 좋아요/스킵 피드백으로 학습되는 재랭킹 모델(지도학습)
- stage1 K-means를 사용자 취향 클러스터링으로 재활용
- 정량 평가(precision@k 등)
- Streamlit Community Cloud 배포
