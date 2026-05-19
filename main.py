import pandas as pd
import numpy as np
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional
from collections import defaultdict
import os
import io
from contextlib import redirect_stderr
from dotenv import load_dotenv, find_dotenv

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    from spotipy.exceptions import SpotifyException

    dotenv_path = find_dotenv('.env', raise_error_if_not_found=False)
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        load_dotenv()

    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    if client_id and client_secret:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))
    else:
        sp = None
except Exception:
    sp = None


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


def build_item_key(name: str, artist: str, year: int) -> str:
    return f"{str(name).strip().lower()}|{str(artist).strip().lower()}|{int(year)}"


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


def find_song(name: str, year: int) -> Optional[pd.DataFrame]:
    if sp is None:
        return None
    
    try:
        results = sp.search(q=f"track:{name} year:{year}", limit=1)
        if not results['tracks']['items']:
            return None

        track = results['tracks']['items'][0]
        track_id = track['id']
        try:
            with redirect_stderr(io.StringIO()):
                audio_features = sp.audio_features([track_id])[0]
        except Exception:
            audio_features = None

        song_data = {
            "name": [name],
            "year": [year],
            "explicit": [int(track['explicit'])],
            "duration_ms": [track['duration_ms']],
            "popularity": [track['popularity']]
        }

        if audio_features is not None:
            for key, value in audio_features.items():
                if value is not None:
                    song_data[key] = [value]

        return pd.DataFrame(song_data)
    except SpotifyException as e:
        print(f"Spotify error for {name} ({year}): {e}")
        return None
    except Exception as e:
        print(f"Error finding song {name} ({year}): {e}")
        return None


def parse_release_year(release_date: str) -> int:
    try:
        return int(str(release_date)[:4])
    except Exception:
        return 0


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
            except Exception:
                audio_features = None
            rows.append(build_spotify_candidate_row(track, audio_features or {}))
        except Exception:
            return

    for song in song_list:
        if song.get('artist'):
            artist_search = sp.search(q=f"artist:{song['artist']}", type='track', limit=10)
            for track in artist_search.get('tracks', {}).get('items', []):
                add_track(track)
                if len(rows) >= max_tracks:
                    break
            if len(rows) >= max_tracks:
                break

        if song.get('name'):
            search_results = sp.search(q=f"track:{song['name']} year:{song.get('year', -1)}", type='track', limit=10)
            for track in search_results.get('tracks', {}).get('items', []):
                add_track(track)
                if len(rows) >= max_tracks:
                    break
        if len(rows) >= max_tracks:
            break

    if not rows:
        return pd.DataFrame(columns=spotify_data.columns)

    return pd.DataFrame(rows[:max_tracks])


def get_mean_vector(song_list: List[Dict], spotify_data: pd.DataFrame) -> Optional[np.ndarray]:
    song_vectors = []
    
    for song in song_list:
        row = find_song_in_data(song.get('name', ''), song.get('year', -1), spotify_data)
        if row is not None:
            vector = row[number_cols].astype(float).values
            song_vectors.append(vector)
            continue

        external = find_song(song.get('name', ''), song.get('year', -1))
        if external is not None:
            try:
                vector = external[number_cols].values[0]
                song_vectors.append(vector)
            except (KeyError, IndexError):
                print(f"Warning: {song.get('name', 'Unknown')} - не всі параметри доступні")
        else:
          
            if song.get('name'):
                row = find_song_in_data(song.get('name', ''), -1, spotify_data)
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
        candidate_data = pd.concat([spotify_data, spotify_candidates], ignore_index=True)
        candidate_data = candidate_data.drop_duplicates(subset=['name', 'artists', 'year']).reset_index(drop=True)

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

    input_mask = np.zeros(len(candidate_data), dtype=bool)
    for song in song_list:
        mask = (candidate_data['name'].str.strip().str.lower() == str(song.get('name', '')).strip().lower()) & \
               (candidate_data['year'] == song.get('year', -1))
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
        name_lower = str(track['name']).strip().lower()
        if name_lower not in seen:
            rec_list.append({
                'name': track['name'],
                'artists': track['artists'],
                'year': int(track['year']),
                'popularity': int(track['popularity']),
                'hybrid_score': round(final_scores[idx], 4),
                'content_similarity': round(content_sim[idx], 4)
            })
            seen.add(name_lower)
            count += 1
    return rec_list


def recommend_songs(
    song_list: List[Dict],
    strategy: str,
    data: pd.DataFrame,
    n_songs: int = 10,
    user_id: Optional[int] = None,
    user_history: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    if user_id is not None and user_history is not None:
        user_recs = recommend_for_user(user_id, data, user_history, n_songs=n_songs)
        if user_recs:
            return user_recs

    if strategy == "Контент тільки":
        return hybrid_recommendations(song_list, data, n_songs=n_songs, alpha=1.0, beta=0.0, gamma=0.0)
    if strategy == "Популярні треки":
        return hybrid_recommendations(song_list, data, n_songs=n_songs, alpha=0.0, beta=1.0, gamma=0.0)
    return hybrid_recommendations(song_list, data, n_songs=n_songs)
