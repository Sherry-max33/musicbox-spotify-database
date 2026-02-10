# etl/etl_load.py
import os
import re
import uuid
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST") or "localhost",
    port=int(os.getenv("DB_PORT") or 5432),
    dbname=os.getenv("DB_NAME") or "musicbox",
    user=os.getenv("DB_USER") or os.getenv("USER"),
    password=os.getenv("DB_PASSWORD") or None,  
)

CSV_PATH = os.path.join("data", "spotify_top_10000.csv")


# ---------- column name normalization ----------
def norm_colname(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace("(ms)", "ms")
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


COLUMN_CANDIDATES = {
    # required
    "track_name": ["track_name", "name", "song", "track"],
    "artist_name": ["artist_name", "artist_name_s", "artist", "artists"],
    "album_name": ["album_name", "album"],
    # ids
    "track_id": ["track_id", "track_uri", "uri", "id"],
    "artist_id": ["artist_id", "artist_uri", "artist_uri_s"],
    "album_id": ["album_id", "album_uri"],
    # optional fields
    "release_date": ["release_date", "album_release_date", "date"],
    "popularity": ["popularity"],
    "duration_ms": ["duration_ms", "track_duration_ms", "duration"],
    "explicit": ["explicit"],
    "preview_url": ["preview_url", "track_preview_url", "preview"],
    "album_image_url": ["album_image_url", "image_url", "cover", "album_cover_url"],
    # genres
    "genres": ["artist_genres", "genres", "genre"],
    # audio features
    "danceability": ["danceability"],
    "energy": ["energy"],
    "valence": ["valence"],
    "acousticness": ["acousticness"],
    "instrumentalness": ["instrumentalness"],
    "liveness": ["liveness"],
    "speechiness": ["speechiness"],
    "tempo": ["tempo"],
    "key": ["key"],
    "mode": ["mode"],
    "loudness": ["loudness"],
    "time_signature": ["time_signature", "timesignature"],
}


def find_col(df: pd.DataFrame, logical_name: str):
    for c in COLUMN_CANDIDATES.get(logical_name, []):
        if c in df.columns:
            return c
    return None


# ---------- helpers ----------
def normalize_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    parts = re.split(r"[:/]", s)
    parts = [p for p in parts if p]
    return parts[-1] if parts else s


def stable_uuid_from_text(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))


def split_multi(val):
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    s = s.replace("，", ",").replace(";", ",").replace("|", ",")
    s = s.strip("[](){}")
    s = s.replace("'", "").replace('"', "")
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_bool(x):
    if x is None or pd.isna(x):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return None


def safe_float(x):
    if x is None or pd.isna(x):
        return None
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def safe_smallint(x, minv=None, maxv=None):
    if x is None or pd.isna(x):
        return None
    try:
        v = int(float(x))
    except Exception:
        return None
    if v in (-1, 999, 9999, 99999):
        return None
    if v < -32768 or v > 32767:
        return None
    if minv is not None and v < minv:
        return None
    if maxv is not None and v > maxv:
        return None
    return v


# ---------- genre grouping (7 big categories) ----------
GENRE_GROUPS = {
    "Pop": ["pop", "k-pop", "kpop", "j-pop", "jpop", "dance pop", "electropop", "indie pop", "teen pop"],
    "Rock": ["rock", "alt rock", "alternative", "indie rock", "hard rock", "classic rock", "punk", "metal", "grunge", "emo"],
    "Hip-Hop": ["hip hop", "hip-hop", "rap", "trap", "drill"],
    "R&B": ["r&b", "rnb", "rhythm and blues", "soul", "neo soul", "funk"],
    "Jazz": ["jazz", "bebop", "swing", "smooth jazz", "bossa nova", "blues"],
    "Classical": ["classical", "orchestra", "orchestral", "symphony", "baroque", "romantic", "opera", "chamber"],
    "Electronic": ["electronic", "edm", "house", "techno", "trance", "dubstep", "drum and bass", "dnb", "ambient", "synthwave", "electro"],
}


def map_genre_to_group(genre: str):
    if not genre:
        return None
    g = genre.strip().lower()
    for group, kws in GENRE_GROUPS.items():
        for kw in kws:
            if kw in g:
                return group
    if "hiphop" in g:
        return "Hip-Hop"
    return None


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = [norm_colname(c) for c in df.columns]

    # Required
    c_track_name = find_col(df, "track_name")
    c_artist_name = find_col(df, "artist_name")
    c_album_name = find_col(df, "album_name")
    if not (c_track_name and c_artist_name and c_album_name):
        raise ValueError(
            "Missing required columns. Need track_name / artist_name / album_name.\n"
            f"Found: track_name={c_track_name}, artist_name={c_artist_name}, album_name={c_album_name}\n"
            f"Available: {list(df.columns)}"
        )

    # IDs
    c_track_id = find_col(df, "track_id")
    c_artist_id = find_col(df, "artist_id")  # artist_uri_s
    c_album_id = find_col(df, "album_id")

    # Optional
    c_release_date = find_col(df, "release_date")
    c_popularity = find_col(df, "popularity")
    c_duration_ms = find_col(df, "duration_ms")
    c_explicit = find_col(df, "explicit")
    c_preview_url = find_col(df, "preview_url")
    c_album_image = find_col(df, "album_image_url")
    c_genres = find_col(df, "genres")

    # Audio cols
    audio_cols = {k: find_col(df, k) for k in [
        "danceability", "energy", "valence", "acousticness", "instrumentalness", "liveness",
        "speechiness", "tempo", "key", "mode", "loudness", "time_signature"
    ]}

    # -----------------------
    # IDs
    # -----------------------
    if c_track_id:
        df["track_id_norm"] = df[c_track_id].apply(normalize_id)
    else:
        df["track_id_norm"] = (
            df[c_track_name].astype(str) + "||" + df[c_artist_name].astype(str) + "||" + df[c_album_name].astype(str)
        ).apply(stable_uuid_from_text)

    if c_album_id:
        df["album_id_norm"] = df[c_album_id].apply(normalize_id)
    else:
        df["album_id_norm"] = (df[c_album_name].astype(str) + "||" + df[c_artist_name].astype(str)).apply(stable_uuid_from_text)

    # -----------------------
    # MULTI-ARTIST EXPANSION
    # -----------------------
    long_rows = []
    for _, r in df.iterrows():
        tid = r["track_id_norm"]
        artist_names = split_multi(r[c_artist_name])
        artist_uris = split_multi(r[c_artist_id]) if c_artist_id else []

        m = max(len(artist_names), len(artist_uris))
        if m == 0:
            continue
        if len(artist_names) < m:
            artist_names += [None] * (m - len(artist_names))
        if len(artist_uris) < m:
            artist_uris += [None] * (m - len(artist_uris))

        for uri, name in zip(artist_uris, artist_names):
            name = None if name is None else str(name).strip()
            uri = None if uri is None else str(uri).strip()

            if uri:
                aid = normalize_id(uri)
            else:
                if not name:
                    continue
                aid = stable_uuid_from_text(name)

            aname = name if name else aid
            long_rows.append((tid, aid, aname))

    track_artist_long = pd.DataFrame(long_rows, columns=["track_id", "artist_id", "artist_name"]).drop_duplicates()
    artists = track_artist_long[["artist_id", "artist_name"]].drop_duplicates()
    track_artist = track_artist_long[["track_id", "artist_id"]].drop_duplicates()

    # -----------------------
    # ALBUMS
    # -----------------------
    albums_cols = ["album_id_norm", c_album_name]
    if c_release_date:
        albums_cols.append(c_release_date)
    if c_album_image:
        albums_cols.append(c_album_image)

    albums = df[albums_cols].drop_duplicates().copy()
    rename_map = {"album_id_norm": "album_id", c_album_name: "album_name"}
    if c_release_date:
        rename_map[c_release_date] = "release_date"
    if c_album_image:
        rename_map[c_album_image] = "album_image_url"
    albums = albums.rename(columns=rename_map)

    # force dates; keep python date or None
    if "release_date" in albums.columns:
        albums["release_date"] = pd.to_datetime(albums["release_date"], errors="coerce").dt.date
        albums["release_date"] = albums["release_date"].where(albums["release_date"].notna(), None)

    # -----------------------
    # TRACKS
    # -----------------------
    tracks_cols = ["track_id_norm", c_track_name]
    if c_popularity:
        tracks_cols.append(c_popularity)
    if c_duration_ms:
        tracks_cols.append(c_duration_ms)
    if c_explicit:
        tracks_cols.append(c_explicit)
    if c_preview_url:
        tracks_cols.append(c_preview_url)

    tracks = df[tracks_cols].drop_duplicates().copy()
    rename_map = {"track_id_norm": "track_id", c_track_name: "track_name"}
    if c_popularity:
        rename_map[c_popularity] = "popularity"
    if c_duration_ms:
        rename_map[c_duration_ms] = "duration_ms"
    if c_explicit:
        rename_map[c_explicit] = "explicit"
    if c_preview_url:
        rename_map[c_preview_url] = "preview_url"
    tracks = tracks.rename(columns=rename_map)

    for c in ["popularity", "duration_ms", "explicit", "preview_url"]:
        if c not in tracks.columns:
            tracks[c] = None

    # -----------------------
    # ALBUM_TRACKS
    # -----------------------
    album_tracks = (
        df[["album_id_norm", "track_id_norm"]]
        .drop_duplicates()
        .rename(columns={"album_id_norm": "album_id", "track_id_norm": "track_id"})
    )
    album_tracks["disc_number"] = 1
    album_tracks["track_number"] = None

    # -----------------------
    # ARTIST_GENRES (fine + 7 groups)
    # -----------------------
    genre_rows = []
    if c_genres:
        track_to_genres = df.set_index("track_id_norm")[c_genres].to_dict()
        for _, ta in track_artist.iterrows():
            tid = ta["track_id"]
            aid = ta["artist_id"]
            gval = track_to_genres.get(tid, None)

            for g in split_multi(gval):
                g_fine = re.sub(r"\s+", " ", g.strip().lower())[:120]
                if not g_fine:
                    continue
                genre_rows.append((aid, g_fine))
                g_group = map_genre_to_group(g_fine)
                if g_group:
                    genre_rows.append((aid, g_group))

    artist_genres = (
        pd.DataFrame(genre_rows, columns=["artist_id", "genre_name"]).drop_duplicates()
        if genre_rows else pd.DataFrame(columns=["artist_id", "genre_name"])
    )

    # -----------------------
    # AUDIO_FEATURES
    # -----------------------
    have_audio = any(v is not None for v in audio_cols.values())
    if have_audio:
        cols = ["track_id_norm"] + [v for v in audio_cols.values() if v is not None]
        af = df[cols].drop_duplicates().copy()
        af = af.rename(columns={"track_id_norm": "track_id"})
        for logical, col in audio_cols.items():
            if col and col in af.columns and col != logical:
                af = af.rename(columns={col: logical})
        audio_features = af
    else:
        audio_features = pd.DataFrame(columns=["track_id"])

    audio_insert_cols = [
        "track_id", "danceability", "energy", "valence", "acousticness", "instrumentalness",
        "liveness", "speechiness", "tempo", "key", "mode", "loudness", "time_signature"
    ]
    for c in audio_insert_cols:
        if c not in audio_features.columns:
            audio_features[c] = None

    # -----------------------
    # LOAD TO POSTGRES
    # -----------------------
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # artists
        execute_values(
            cur,
            "INSERT INTO artists (artist_id, artist_name, status) VALUES %s "
            "ON CONFLICT (artist_id) DO UPDATE SET artist_name=EXCLUDED.artist_name",
            [(a, n, "approved") for a, n in artists.itertuples(index=False, name=None)],
            page_size=2000
        )

        # albums (robust NaT/"NaT" -> None)
        album_rows = []
        for _, r in albums.iterrows():
            rd = r["release_date"] if "release_date" in albums.columns else None
            if rd is None or pd.isna(rd) or str(rd).strip().lower() == "nat":
                rd = None

            img = r["album_image_url"] if "album_image_url" in albums.columns else None
            if img is None or pd.isna(img) or str(img).strip().lower() in ("nan", "none"):
                img = None

            album_rows.append((r["album_id"], r["album_name"], rd, img, "approved"))

        execute_values(
            cur,
            "INSERT INTO albums (album_id, album_name, release_date, album_image_url, status) VALUES %s "
            "ON CONFLICT (album_id) DO UPDATE SET album_name=EXCLUDED.album_name",
            album_rows,
            page_size=2000
        )

        # tracks
        track_rows = []
        for _, r in tracks.iterrows():
            pop = None if pd.isna(r["popularity"]) else int(r["popularity"])
            dur = None if pd.isna(r["duration_ms"]) else int(r["duration_ms"])
            exp = parse_bool(r["explicit"])
            prv = None if pd.isna(r["preview_url"]) else str(r["preview_url"])
            track_rows.append((r["track_id"], r["track_name"], pop, dur, exp, prv, "approved"))

        execute_values(
            cur,
            "INSERT INTO tracks (track_id, track_name, popularity, duration_ms, explicit, preview_url, status) VALUES %s "
            "ON CONFLICT (track_id) DO UPDATE SET track_name=EXCLUDED.track_name",
            track_rows,
            page_size=2000
        )

        # track_artist
        execute_values(
            cur,
            "INSERT INTO track_artist (track_id, artist_id) VALUES %s ON CONFLICT DO NOTHING",
            list(track_artist.itertuples(index=False, name=None)),
            page_size=5000
        )

        # album_tracks
        execute_values(
            cur,
            "INSERT INTO album_tracks (album_id, track_id, disc_number, track_number) VALUES %s ON CONFLICT DO NOTHING",
            list(album_tracks.itertuples(index=False, name=None)),
            page_size=5000
        )

        # artist_genres
        if len(artist_genres) > 0:
            execute_values(
                cur,
                "INSERT INTO artist_genres (artist_id, genre_name) VALUES %s ON CONFLICT DO NOTHING",
                list(artist_genres.itertuples(index=False, name=None)),
                page_size=5000
            )

        # audio_features (FINAL SAFETY: build python-native rows, prevent smallint overflow)
        if have_audio and len(audio_features) > 0:
            audio_rows = []
            for _, r in audio_features.iterrows():
                audio_rows.append((
                    r.get("track_id"),
                    safe_float(r.get("danceability")),
                    safe_float(r.get("energy")),
                    safe_float(r.get("valence")),
                    safe_float(r.get("acousticness")),
                    safe_float(r.get("instrumentalness")),
                    safe_float(r.get("liveness")),
                    safe_float(r.get("speechiness")),
                    safe_float(r.get("tempo")),
                    safe_smallint(r.get("key"), minv=0, maxv=11),
                    safe_smallint(r.get("mode"), minv=0, maxv=1),
                    safe_float(r.get("loudness")),
                    safe_smallint(r.get("time_signature"), minv=0, maxv=16),
                ))

            execute_values(
                cur,
                "INSERT INTO audio_features (track_id, danceability, energy, valence, acousticness, instrumentalness, "
                "liveness, speechiness, tempo, key, mode, loudness, time_signature) VALUES %s "
                "ON CONFLICT (track_id) DO NOTHING",
                audio_rows,
                page_size=3000
            )

        conn.commit()
        print("✅ ETL completed.")
        print(f"Artists: {len(artists)} | Albums: {len(albums)} | Tracks: {len(tracks)}")
        print(f"Track_Artist: {len(track_artist)} | Album_Tracks: {len(album_tracks)} | Artist_Genres(rows): {len(artist_genres)}")
        print(f"Audio_Features: {len(audio_features) if have_audio else 0}")

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
