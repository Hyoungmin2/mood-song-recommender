"""
Spotify Web API 클라이언트 (Client Credentials Flow)

- 사용자 로그인 없이 앱 단위로 공개 카탈로그(검색, 아티스트, 트랙)에 접근.
- 2024년 11월 27일 이후 audio-features / audio-analysis / recommendations /
  related-artists 엔드포인트는 신규 앱에서 deprecated 되어 403을 반환하므로
  이 프로젝트에서는 사용하지 않음. 대신 search / artists 메타데이터만 활용하고,
  오디오 특징은 Kaggle 데이터셋(build_dataset.py)에서 가져온다.
"""
import os
import time
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")


def get_spotify_client():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET이 설정되지 않음. "
            ".env 파일을 만들고 .env.example을 참고해서 키를 채워넣을 것."
        )
    auth_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def search_track(sp, title, artist="", retries=3):
    """곡 제목(+아티스트)으로 Spotify에서 검색해 메타데이터 반환. 못 찾으면 None."""
    query = f"track:{title}" + (f" artist:{artist}" if artist else "")
    for attempt in range(retries):
        try:
            result = sp.search(q=query, type="track", limit=1)
            items = result.get("tracks", {}).get("items", [])
            if not items:
                return None
            track = items[0]
            images = track["album"].get("images", [])
            return {
                "spotify_id": track["id"],
                "spotify_name": track["name"],
                "spotify_artist": track["artists"][0]["name"],
                "album_art_url": images[0]["url"] if images else None,
                "popularity": track["popularity"],
                "preview_url": track.get("preview_url"),
                "external_url": track["external_urls"].get("spotify"),
            }
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:  # rate limit
                wait = int(e.headers.get("Retry-After", 1))
                time.sleep(wait)
                continue
            return None
        except Exception:
            time.sleep(1)
    return None


def get_artist_genres(sp, artist_name):
    """아티스트 이름으로 검색해 장르 태그 목록 가져오기."""
    result = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
    items = result.get("artists", {}).get("items", [])
    if not items:
        return []
    return items[0].get("genres", [])