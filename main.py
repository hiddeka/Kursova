import pandas as pd
import numpy as np
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional
from collections import defaultdict
import os
import io
import traceback
import hashlib
import re
from contextlib import redirect_stderr
from dotenv import load_dotenv, find_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


load_dotenv()

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

number_cols = [
    'valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
    'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
]


@lru_cache(maxsize=1)
def load_data(path: str = "data.csv") -> pd.DataFrame:
    data = pd.read_csv(path)
    data["name"] = data["name"].astype(str)
    data["artists"] = data["artists"].astype(str)
    data["year"] = pd.to_numeric(data["year"], errors="coerce").fillna(0).astype(int)
    return data


@lru_cache(maxsize=1)
def load_user_history(path: str = "user_favorites.csv") -> pd.DataFrame:
    history = pd.read_csv(path)
    history["user_id"] = pd.to_numeric(history["user_id"], errors="coerce").fillna(0).astype(int)
    history["name"] = history["name"].astype(str)
    history["artist"] = history["artist"].astype(str)
    history["year"] = pd.to_numeric(history["year"], errors="coerce").fillna(0).astype(int)
    if "rating" not in history.columns:
        history["rating"] = 1.0
    history["rating"] = pd.to_numeric(history["rating"], errors="coerce").fillna(1.0).clip(1.0, 5.0)
    return history


def load_users(path: str = "users.csv") -> pd.DataFrame:
    users = pd.read_csv(path)
    users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce").fillna(0).astype(int)
    users["name"] = users["name"].astype(str)
    if "password_hash" not in users.columns:
        users["password_hash"] = ""
    users["password_hash"] = users["password_hash"].fillna("").astype(str)
    return users


def hash_password(password: str) -> str:
    return hashlib.sha256(str(password).encode("utf-8")).hexdigest()


def register_user(name: str, password: str, path: str = "users.csv") -> Dict:
    name = str(name).strip()
    password = str(password)
    if not name or not password:
        return {"ok": False, "message": "Введіть ім'я користувача і пароль."}

    users = load_users(path)
    if users["name"].str.strip().str.lower().eq(name.lower()).any():
        return {"ok": False, "message": "Користувач з таким ім'ям вже існує."}

    next_id = int(users["user_id"].max()) + 1 if not users.empty else 1
    new_user = pd.DataFrame([{
        "user_id": next_id,
        "name": name,
        "password_hash": hash_password(password),
    }])
    users = pd.concat([users, new_user], ignore_index=True)
    users[["user_id", "name", "password_hash"]].to_csv(path, index=False)
    load_data.cache_clear()
    load_user_history.cache_clear()
    return {"ok": True, "message": "Користувача зареєстровано.", "user_id": next_id, "name": name}


def authenticate_user(name: str, password: str, path: str = "users.csv") -> Optional[Dict]:
    users = load_users(path)
    name_key = str(name).strip().lower()
    matched = users[users["name"].str.strip().str.lower() == name_key]
    if matched.empty:
        return None

    user = matched.iloc[0]
    stored_hash = str(user.get("password_hash", ""))
    if stored_hash and stored_hash != hash_password(password):
        return None
    if not stored_hash and password:
        return None

    return {"user_id": int(user["user_id"]), "name": user["name"]}


def save_user_rating(
    user_id: int,
    track: Dict,
    rating: float,
    path: str = "user_favorites.csv"
) -> Dict:
    rating = float(rating)
    rating = min(max(rating, 1.0), 5.0)

    history = load_user_history(path)
    name = str(track.get("name", "")).strip()
    artist = str(track.get("artist") or track.get("artists", "")).strip()
    year = int(track.get("year", 0) or 0)
    if not name:
        return {"ok": False, "message": "Немає назви треку для оцінки."}

    mask = (
        (history["user_id"] == int(user_id))
        & (history["name"].str.strip().str.lower() == name.lower())
        & (history["artist"].str.strip().str.lower() == artist.lower())
        & (history["year"] == year)
    )

    if mask.any():
        history.loc[mask, "rating"] = rating
    else:
        history = pd.concat([history, pd.DataFrame([{
            "user_id": int(user_id),
            "name": name,
            "artist": artist,
            "year": year,
            "rating": rating,
        }])], ignore_index=True)

    history[["user_id", "name", "artist", "year", "rating"]].to_csv(path, index=False)
    load_user_history.cache_clear()
    return {"ok": True, "message": "Оцінку збережено."}


def get_user_ratings(user_id: int, path: str = "user_favorites.csv") -> pd.DataFrame:
    history = load_user_history(path)
    return history[history["user_id"] == int(user_id)].copy()


def build_item_key(name: str, artist: str, year: int) -> str:
    return f"{str(name).strip().lower()}|{str(artist).strip().lower()}|{int(year)}"


def normalize_track_name(name: str) -> str:
    return re.sub(r"[^a-zа-яіїєґ0-9]+", "", str(name).strip().lower())


def filter_rated_tracks(recommendations: List[Dict], rated_tracks: List[Dict]) -> List[Dict]:
    rated_names = {
        normalize_track_name(track.get("name", ""))
        for track in rated_tracks
        if normalize_track_name(track.get("name", ""))
    }
    if not rated_names:
        return recommendations

    filtered = []
    for recommendation in recommendations:
        if normalize_track_name(recommendation.get("name", "")) not in rated_names:
            filtered.append(recommendation)
    return filtered


def ensure_search_columns(data: pd.DataFrame) -> pd.DataFrame:
    if "_name_norm" not in data.columns:
        data["_name_norm"] = data["name"].astype(str).str.strip().str.lower()
    if "_artists_norm" not in data.columns:
        data["_artists_norm"] = data["artists"].astype(str).str.strip().str.lower()
    return data


def find_user_history_item(row: pd.Series, data: pd.DataFrame) -> Optional[pd.Series]:
    matched = find_song_in_data(row["name"], row["year"], data, row.get("artist"))
    if matched is not None:
        return matched
    return find_song_in_data(row["name"], -1, data, row.get("artist"))


def build_user_history_items(user_history: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    data = ensure_search_columns(data)
    rows = []
    for _, row in user_history.iterrows():
        matched = find_user_history_item(row, data)
        if matched is None:
            continue

        item_key = build_item_key(matched["name"], matched["artists"], matched["year"])
        rows.append({
            "user_id": int(row["user_id"]),
            "item_key": item_key,
            "name": matched["name"],
            "artists": matched["artists"],
            "year": int(matched["year"]),
            "popularity": int(matched.get("popularity", 0)),
            "rating": float(row.get("rating", 1.0)),
        })

    if not rows:
        return pd.DataFrame(columns=["user_id", "item_key", "name", "artists", "year", "popularity", "rating"])
    return pd.DataFrame(rows)


def build_user_item_matrix(user_history: pd.DataFrame, data: pd.DataFrame):
    history_items = build_user_history_items(user_history, data)
    if history_items.empty:
        return None, None

    user_item = history_items.pivot_table(
        index="user_id",
        columns="item_key",
        values="rating",
        fill_value=0.0,
    )

    item_info = (
        history_items
        .drop_duplicates(subset=["item_key"])
        .set_index("item_key")[['name', 'artists', 'year', 'popularity']]
    )
    return user_item, item_info


def user_user_cf_recommendations(
    user_id: int,
    user_history: pd.DataFrame,
    data: pd.DataFrame,
    n_songs: int = 10,
    top_k: int = 3,
) -> List[Dict]:
    user_item, item_info = build_user_item_matrix(user_history, data)
    if user_item is None or user_id not in user_item.index:
        return []

    target_vector = user_item.loc[user_id].values
    if target_vector.sum() == 0:
        return []

    similarity_matrix = cosine_similarity(user_item.values)
    user_index = list(user_item.index).index(user_id)
    similarity_scores = similarity_matrix[user_index]
    similarity_scores[user_index] = 0.0

    positive_indices = np.where(similarity_scores > 0)[0]
    if positive_indices.size == 0:
        return []

    neighbor_indices = positive_indices[np.argsort(similarity_scores[positive_indices])[::-1][:top_k]]
    weights = similarity_scores[neighbor_indices]
    if weights.sum() == 0:
        return []

    neighbor_matrix = user_item.values[neighbor_indices]
    predicted_scores = np.dot(weights, neighbor_matrix) / (np.sum(np.abs(weights)) + 1e-9)

    candidate_mask = target_vector == 0
    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.size == 0:
        return []

    recommendations = []
    for idx in candidate_indices[np.argsort(predicted_scores[candidate_indices])[::-1]]:
        score = float(predicted_scores[idx])
        if score <= 0:
            continue
        item_key = user_item.columns[idx]
        info = item_info.loc[item_key]
        recommendations.append({
            "name": info["name"],
            "artists": info["artists"],
            "year": int(info["year"]),
            "score": round(score, 4),
            "popularity": int(info["popularity"]),
        })
        if len(recommendations) >= n_songs:
            break

    return recommendations


def recommend_for_user(
    user_id: int,
    data: pd.DataFrame,
    user_history: pd.DataFrame,
    n_songs: int = 10,
) -> List[Dict]:
    return user_user_cf_recommendations(user_id, user_history, data, n_songs)


def find_song(name: str, year: int, artist: Optional[str] = None) -> Optional[pd.DataFrame]:
    if sp is None:
        return None

    try:
        query_parts = [f"track:{name}"]
        if artist:
            query_parts.append(f"artist:{artist}")
        if year and year > 0:
            query_parts.append(f"year:{year}")

        results = sp.search(q=" ".join(query_parts), type="track", limit=1)
        if not results['tracks']['items']:
            print(f"Spotify track not found for {name} ({year})")
            return None

        track = results['tracks']['items'][0]
        track_id = track['id']
        with redirect_stderr(io.StringIO()):
            audio_features = sp.audio_features([track_id])[0]
    except SpotifyException:
        traceback.print_exc()
        return None
    except Exception:
        traceback.print_exc()
        return None

    song_data = {
        "name": [track.get('name', name)],
        "artists": [', '.join(artist.get('name', '') for artist in track.get('artists', []))],
        "year": [parse_release_year(track.get('album', {}).get('release_date', '')) or year],
        "explicit": [int(track['explicit'])],
        "duration_ms": [track['duration_ms']],
        "popularity": [track['popularity']]
    }

    if audio_features is not None:
        for key, value in audio_features.items():
            if value is not None:
                song_data[key] = [value]

    return pd.DataFrame(song_data)


def parse_release_year(release_date: str) -> int:
    try:
        return int(str(release_date)[:4])
    except Exception:
        return 0


def search_tracks(query: str, data: pd.DataFrame, limit: int = 10) -> List[Dict]:
    query = str(query).strip()
    if not query:
        return []

    results = []
    seen = set()

    if sp is not None:
        try:
            spotify_results = sp.search(q=query, type="track", limit=limit)
            for track in spotify_results.get("tracks", {}).get("items", []):
                artists = ", ".join(
                    artist.get("name", "")
                    for artist in track.get("artists", [])
                    if artist.get("name")
                )
                row = {
                    "name": track.get("name", ""),
                    "artist": artists,
                    "year": parse_release_year(track.get("album", {}).get("release_date", "")),
                    "popularity": int(track.get("popularity", 0)),
                    "source": "Spotify",
                }
                key = build_item_key(row["name"], row["artist"], row["year"])
                if row["name"] and key not in seen:
                    seen.add(key)
                    results.append(row)
        except SpotifyException:
            traceback.print_exc()
        except Exception:
            traceback.print_exc()

    if results:
        return results[:limit]

    data = ensure_search_columns(data)
    key = query.lower()
    matched = data[
        data["_name_norm"].str.contains(key, regex=False, na=False)
        | data["_artists_norm"].str.contains(key, regex=False, na=False)
    ]
    if "popularity" in matched.columns:
        matched = matched.sort_values("popularity", ascending=False)

    for _, row in matched.head(limit).iterrows():
        artists = str(row.get("artists", ""))
        result = {
            "name": str(row.get("name", "")),
            "artist": artists,
            "year": int(row.get("year", 0)),
            "popularity": int(row.get("popularity", 0)),
            "source": "data.csv",
        }
        result_key = build_item_key(result["name"], result["artist"], result["year"])
        if result["name"] and result_key not in seen:
            seen.add(result_key)
            results.append(result)

    return results[:limit]


def get_spotify_track_id(name: str, year: int) -> Optional[str]:
    if sp is None:
        return None

    try:
        results = sp.search(q=f"track:{name} year:{year}", type="track", limit=1)
        items = results.get('tracks', {}).get('items', [])
        if not items:
            return None
        return items[0]['id']
    except Exception:
        return None


def get_spotify_artist_id(artist_name: str) -> Optional[str]:
    if sp is None:
        return None

    try:
        results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
        items = results.get('artists', {}).get('items', [])
        if not items:
            return None
        return items[0]['id']
    except Exception:
        return None


def build_spotify_candidate_row(track: Dict, audio_features: Dict) -> Dict:
    artist_names = [artist.get('name', '') for artist in track.get('artists', [])]
    artists = ', '.join(filter(None, artist_names))
    release_date = track.get('album', {}).get('release_date', '')
    year = parse_release_year(release_date)

    row = {
        'name': track.get('name', ''),
        'artists': artists,
        'year': year,
        'explicit': int(track.get('explicit', False)),
        'duration_ms': int(track.get('duration_ms', 0)),
        'popularity': int(track.get('popularity', 0)),
    }
    for col in number_cols:
        row[col] = float(audio_features.get(col, 0.0)) if audio_features.get(col) is not None else 0.0
    return row


def find_song_in_data(name: str, year: int, data: pd.DataFrame, artist: Optional[str] = None) -> Optional[pd.Series]:
    data = ensure_search_columns(data)
    key = str(name).strip().lower()
    if not key:
        return None

    artist_key = str(artist or "").strip().lower()

    def best_match(candidates: pd.DataFrame) -> Optional[pd.Series]:
        if candidates.empty:
            return None
        if artist_key:
            artist_matches = candidates[
                candidates['_artists_norm'].str.contains(artist_key, regex=False, na=False)
            ]
            if artist_matches.empty:
                return None
            candidates = artist_matches
        if 'popularity' in candidates.columns:
            candidates = candidates.sort_values('popularity', ascending=False)
        return candidates.iloc[0]

    exact = data[
        (data['_name_norm'] == key)
        & (data['year'] == year)
    ]
    matched = best_match(exact)
    if matched is not None:
        return matched

    contains_with_year = data[
        (data['_name_norm'].str.contains(key, regex=False, na=False))
        & (data['year'] == year)
    ]
    matched = best_match(contains_with_year)
    if matched is not None:
        return matched

    name_only = data[data['_name_norm'] == key]
    matched = best_match(name_only)
    if matched is not None:
        return matched

    contains = data[data['_name_norm'].str.contains(key, regex=False, na=False)]
    matched = best_match(contains)
    if matched is not None:
        return matched

    return None


def get_spotify_candidate_pool(song_list: List[Dict], spotify_data: pd.DataFrame, max_tracks: int = 120) -> pd.DataFrame:
    if sp is None:
        return pd.DataFrame(columns=spotify_data.columns)

    rows = []
    seen_ids = set()

    def add_track(track: Dict):
        try:
            track_id = track.get('id')
            if not track_id or track_id in seen_ids:
                return
            seen_ids.add(track_id)
            try:
                with redirect_stderr(io.StringIO()):
                    audio_features = sp.audio_features([track_id])[0]
            except SpotifyException:
                traceback.print_exc()
                audio_features = None
            except Exception:
                traceback.print_exc()
                audio_features = None
            rows.append(build_spotify_candidate_row(track, audio_features or {}))
        except Exception:
            return

    for song in song_list:
        try:
            if song.get('artist'):
                artist_search = sp.search(q=f"artist:{song['artist']}", type='track', limit=10)
                for track in artist_search.get('tracks', {}).get('items', []):
                    add_track(track)
                    if len(rows) >= max_tracks:
                        break
                if len(rows) >= max_tracks:
                    break

            if song.get('name'):
                query_parts = [f"track:{song['name']}"]
                if song.get('artist'):
                    query_parts.append(f"artist:{song['artist']}")
                if song.get('year') and song.get('year', -1) > 0:
                    query_parts.append(f"year:{song['year']}")
                search_results = sp.search(q=" ".join(query_parts), type='track', limit=10)
                for track in search_results.get('tracks', {}).get('items', []):
                    add_track(track)
                    if len(rows) >= max_tracks:
                        break
        except SpotifyException:
            traceback.print_exc()
        except Exception:
            traceback.print_exc()
        if len(rows) >= max_tracks:
            break

    if not rows:
        return pd.DataFrame(columns=spotify_data.columns)

    return pd.DataFrame(rows[:max_tracks])


def get_mean_vector(song_list: List[Dict], spotify_data: pd.DataFrame) -> Optional[np.ndarray]:
    song_vectors = []
    
    for song in song_list:
        external = find_song(
            song.get('name', ''),
            song.get('year', -1),
            song.get('artist')
        )
        if external is not None:
            try:
                vector = external[number_cols].values[0]
                song_vectors.append(vector)
                continue
            except (KeyError, IndexError):
                external = None
                print(f"Warning: {song.get('name', 'Unknown')} - не всі параметри доступні")
        if external is None:
          
            if song.get('name'):
                row = find_song_in_data(
                    song.get('name', ''),
                    song.get('year', -1),
                    spotify_data,
                    song.get('artist')
                )
                if row is None:
                    row = find_song_in_data(song.get('name', ''), -1, spotify_data, song.get('artist'))
                if row is not None:
                    vector = row[number_cols].astype(float).values
                    song_vectors.append(vector)
                else:
                    print(f"Warning: {song.get('name', 'Unknown')} not found")

    if len(song_vectors) == 0:
        return None
    
    return np.mean(song_vectors, axis=0)


def hybrid_recommendations(
    song_list: List[Dict],
    spotify_data: pd.DataFrame,
    n_songs: int = 10,
    alpha: float = 0.40,
    beta: float = 0.30,
    gamma: float = 0.15
) -> List[Dict]:
    if not song_list:
        return []
    
    song_center = get_mean_vector(song_list, spotify_data)
    if song_center is None:
        return []
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(spotify_data[number_cols].fillna(0).values)
    scaled_center = scaler.transform(song_center.reshape(1, -1))
    
    spotify_candidates = get_spotify_candidate_pool(song_list, spotify_data, max_tracks=120)
    candidate_data = spotify_data
    if not spotify_candidates.empty:
        candidate_data = spotify_candidates.drop_duplicates(subset=['name', 'artists', 'year']).reset_index(drop=True)

    scaled_features = scaler.fit_transform(candidate_data[number_cols].fillna(0).values)
    scaled_center = scaler.transform(song_center.reshape(1, -1))
    content_sim = cosine_similarity(scaled_center, scaled_features)[0]

    popular_idx = candidate_data.nlargest(8000, 'popularity').index
    reduced_features = scaled_features[popular_idx]
    reduced_sim = cosine_similarity(scaled_center, reduced_features)[0]

    collab_scores = np.zeros(len(candidate_data))
    collab_scores[popular_idx] = reduced_sim

    pop_scores = MinMaxScaler().fit_transform(
        candidate_data['popularity'].values.reshape(-1, 1)
    ).flatten()

    final_scores = (alpha * content_sim + beta * collab_scores + gamma * pop_scores)

    rated_names = {
        normalize_track_name(song.get('name', ''))
        for song in song_list
        if normalize_track_name(song.get('name', ''))
    }

    input_mask = np.zeros(len(candidate_data), dtype=bool)
    for song in song_list:
        song_name = normalize_track_name(song.get('name', ''))
        mask = candidate_data['name'].apply(normalize_track_name) == song_name
        input_mask |= mask.values

    final_scores[input_mask] = -9999

    final_scores[content_sim > 0.98] -= 10.0
    final_scores[content_sim > 0.96] -= 5.0
    final_scores[content_sim > 0.94] -= 2.0

    recommended_idx = np.argsort(final_scores)[::-1]
    
    rec_list = []
    seen = set()
    count = 0
    for idx in recommended_idx:
        if count >= n_songs:
            break
        track = candidate_data.iloc[idx]
        name_key = normalize_track_name(track['name'])
        if name_key in rated_names or final_scores[idx] <= -9999:
            continue
        if name_key not in seen:
            rec_list.append({
                'name': track['name'],
                'artists': track['artists'],
                'year': int(track['year']),
                'popularity': int(track['popularity']),
                'hybrid_score': round(final_scores[idx], 4),
                'content_similarity': round(content_sim[idx], 4)
            })
            seen.add(name_key)
            count += 1
    return rec_list


def recommend_songs(
    song_list: List[Dict],
    strategy: str,
    data: pd.DataFrame,
    n_songs: int = 30,
    user_id: Optional[int] = None,
    user_history: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    recommendations = []

    if strategy == "User-based CF" and user_id is not None and user_history is not None:
        user_recs = recommend_for_user(user_id, data, user_history, n_songs=n_songs)
        if user_recs:
            recommendations = user_recs

    if not recommendations:
        if strategy == "Контент тільки":
            recommendations = hybrid_recommendations(
                song_list, data, n_songs=n_songs, alpha=1.0, beta=0.0, gamma=0.0
            )
        elif strategy == "Популярні треки":
            recommendations = hybrid_recommendations(
                song_list, data, n_songs=n_songs, alpha=0.0, beta=1.0, gamma=0.0
            )
        else:
            recommendations = hybrid_recommendations(song_list, data, n_songs=n_songs)

    return filter_rated_tracks(recommendations, song_list)[:n_songs]
