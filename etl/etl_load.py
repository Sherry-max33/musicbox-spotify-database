# etl/etl_load.py
"""
ETL: Load Spotify (and similar) data into the musicbox database.

Genre mapping (Big-8)
--------------------
Fine genres from the source are mapped to 8 main categories (Big-8) and written to artist_genres (fine genres are kept).
Viewer's EXPLORE BY GENRES filters by these 8 categories.

- Matching: for each fine-genre string (strip + lower), if it "contains" any keyword of a category, assign that category.
- Priority: GENRE_GROUPS dict order; first match wins (Country → Pop → Rock → …).
- Exceptions:
  - "hiphop" (no hyphen) is treated as Hip-Hop.
  - Electronic: exclude "edm" and generic "house" to avoid pop/dance artists being classified as electronic; if fine genre is tropical / dance pop / pop dance / edm, do not assign Electronic even if other keywords match.

8 categories and keywords (see GENRE_GROUPS below):
  Country, Pop, Rock, Hip-Hop, R&B, Jazz, Classical, Electronic.

Primary genre for songs/albums/artists (track_genres + artist_primary_genre)
----------------------------------------------------------------------------
- Songs/albums: only the "first" genre in the CSV track genre list is mapped to Big-8 and written to track_genres (one row per track). Viewer EXPLORE BY GENRES uses this.
- Artists: same as song/album—one primary Big-8 per artist: primary genre of that artist's "highest popularity" track that has track_genres, written to artist_primary_genre (one row per artist). Viewer EXPLORE BY GENRES artist tab uses this.
- Example: "dance pop, pop, urban contemporary, r&b" → take only "dance pop" → Pop; that track/artist under that track appears only under Pop.
- artist_genres is still written (all fine genres mapped) for Analyst etc.; EXPLORE BY GENRES artist filter uses artist_primary_genre.
"""
import argparse
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
    "release_date": ["album_release_date", "release_date", "date"],
    "popularity": ["popularity"],
    "duration_ms": ["duration_ms", "track_duration_ms", "duration"],
    "explicit": ["explicit"],
    "preview_url": ["preview_url", "track_preview_url", "preview"],
    "album_image_url": ["album_image_url", "image_url", "cover", "album_cover_url"],
    "added_at": ["added_at"],
    # disc / track numbers within album
    "disc_number": ["disc_number", "disc_num"],
    "track_number": ["track_number", "track_num"],
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


def normalize_release_date_str(x):
    """Normalize inconsistent release date strings for parsing: strip quotes, normalize/remove dashes, keep only digits and standard separators; then parse and take year."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Strip leading/trailing quotes and backticks
    s = re.sub(r"^['\"`\s]+|['\"`\s]+$", "", s)
    if not s or s.lower() in ("nan", "none", "nat", ""):
        return ""
    # Normalize dashes: Unicode hyphen, en-dash, em-dash etc. to hyphen or remove (parse year from digits only if needed)
    s = re.sub(r"[\u2010-\u2015\u2212\uff0d\-–—]", "-", s)
    # Remove extra spaces
    s = re.sub(r"\s+", "", s)
    # Slash to hyphen (e.g. 2003/01/14)
    s = s.replace("/", "-")
    # Keep only digits and hyphen to avoid parse failures
    s = re.sub(r"[^\d\-]", "", s)
    return s


def parse_release_date_to_year(s: str):
    """Parse release date string; return (year, date) or (None, None). Store as date(year, 1, 1) after normalizing to year."""
    s = normalize_release_date_str(s)
    if not s:
        return None, None
    try:
        # Support 2009, 2003-01-14, 20030114, etc.
        if re.match(r"^\d{4}$", s):
            y = int(s)
            if 1900 <= y <= 2100:
                return y, pd.Timestamp(year=y, month=1, day=1).date()
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None, None
        y = dt.year
        if 1900 <= y <= 2100:
            return y, pd.Timestamp(year=y, month=1, day=1).date()
    except Exception:
        pass
    return None, None


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


# ---------- genre grouping (Big-8) ----------
# Rule: fine-genre string containing any keyword of a category assigns that category; order = priority. Country first so "country pop" etc. map to Country.
GENRE_GROUPS = {
    "Country": ["country"],
    "Pop": ["pop", "k-pop", "kpop", "j-pop", "jpop", "dance pop", "electropop", "indie pop", "teen pop"],
    "Rock": ["rock", "alt rock", "alternative", "indie rock", "hard rock", "classic rock", "punk", "metal", "grunge", "emo"],
    "Hip-Hop": ["hip hop", "hip-hop", "rap", "trap", "drill"],
    "R&B": ["r&b", "rnb", "rhythm and blues", "soul", "neo soul", "funk"],
    "Jazz": ["jazz", "bebop", "swing", "smooth jazz", "bossa nova", "blues"],
    "Classical": ["classical", "orchestra", "orchestral", "symphony", "baroque", "romantic", "opera", "chamber"],
    # Exclude "edm": many pop artists are labeled edm in source and would be misclassified as electronic; exclude generic "house" to avoid tropical house / dance pop etc.
    "Electronic": ["electronic", "techno", "trance", "dubstep", "drum and bass", "dnb", "ambient", "synthwave", "electro", "electro house", "filter house", "progressive house"],
}


def map_genre_to_group(genre: str):
    if not genre:
        return None
    g = genre.strip().lower()
    for group, kws in GENRE_GROUPS.items():
        for kw in kws:
            if kw in g:
                # Avoid tropical house / dance etc. putting pop artists in Electronic
                if group == "Electronic" and ("tropical" in g or g in ("dance pop", "pop dance", "edm")):
                    continue
                return group
    if "hiphop" in g:
        return "Hip-Hop"
    return None


def main():
    parser = argparse.ArgumentParser(description="ETL: load Spotify data into musicbox DB")
    parser.add_argument("--albums-only", action="store_true", help="Reload only albums (TRUNCATE albums + album_tracks, then insert corrected data)")
    parser.add_argument("--verify", action="store_true", help="Run verification query: albums total, null release_date count, pct_null")
    parser.add_argument("--print-columns", action="store_true", help="Print all column names after ETL (no DB write)")
    args = parser.parse_args()
    albums_only = args.albums_only
    do_verify = args.verify
    do_print_columns = args.print_columns

    if do_verify:
        conn = psycopg2.connect(**DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT
              COUNT(*) AS total,
              COUNT(*) FILTER (WHERE release_date IS NULL) AS null_cnt,
              ROUND(100.0 * COUNT(*) FILTER (WHERE release_date IS NULL) / NULLIF(COUNT(*), 0), 2) AS pct_null
            FROM albums;
        """)
        row = cur.fetchone()
        print(f"Albums: total={row[0]}, null release_date={row[1]}, pct_null={row[2]}%")
        cur.close()
        conn.close()
        return

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    # Strip BOM and whitespace from column names so "Album Release Date" is found
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    # Use exact source column "Album Release Date" for album release date (before normalizing)
    for c in list(df.columns):
        if c.strip() == "Album Release Date":
            df = df.rename(columns={c: "album_release_date"})
            break
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

    # Optional: release_date mapping chain
    # Source CSV column "Album Release Date" -> renamed to "album_release_date" -> find_col yields c_release_date
    # -> albums_cols includes c_release_date -> renamed to "release_date" -> DB albums.release_date
    c_release_date = find_col(df, "release_date")
    c_popularity = find_col(df, "popularity")
    c_duration_ms = find_col(df, "duration_ms")
    c_explicit = find_col(df, "explicit")
    c_preview_url = find_col(df, "preview_url")
    c_album_image = find_col(df, "album_image_url")
    c_added_at = find_col(df, "added_at")
    c_genres = find_col(df, "genres")
    c_disc_number = find_col(df, "disc_number")
    c_track_number = find_col(df, "track_number")

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

    # One row per album: when multiple rows per album, keep row with non-null release_date (sort by album then date, nulls last)
    _albums_df = df[albums_cols].copy()
    if c_release_date:
        # Normalize via regex: strip quotes, normalize/remove dash; after parse store as date(year, 1, 1)
        _albums_df["_rd"] = _albums_df[c_release_date].apply(
            lambda x: parse_release_date_to_year(x)[1]
        )
        _albums_df["_rd_ts"] = pd.to_datetime(_albums_df["_rd"], errors="coerce")
        _albums_df = _albums_df.sort_values(["album_id_norm", "_rd_ts"], na_position="last")
        _albums_df = _albums_df.drop(columns=["_rd_ts"])
    albums = _albums_df.drop_duplicates(subset=["album_id_norm"], keep="first").copy()
    if c_release_date:
        albums["release_date"] = albums["_rd"]
        albums["release_date"] = albums["release_date"].where(
            albums["release_date"].notna(), None
        )
        albums = albums.drop(columns=["_rd", c_release_date])
    rename_map = {"album_id_norm": "album_id", c_album_name: "album_name"}
    if c_album_image:
        rename_map[c_album_image] = "album_image_url"
    albums = albums.rename(columns=rename_map)

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
    if c_added_at:
        tracks_cols.append(c_added_at)

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
    if c_added_at:
        rename_map[c_added_at] = "added_at"
    tracks = tracks.rename(columns=rename_map)

    for c in ["popularity", "duration_ms", "explicit", "preview_url", "added_at"]:
        if c not in tracks.columns:
            tracks[c] = None

    # -----------------------
    # ALBUM_TRACKS
    # -----------------------
    at_cols = ["album_id_norm", "track_id_norm"]
    if c_disc_number:
        at_cols.append(c_disc_number)
    if c_track_number:
        at_cols.append(c_track_number)

    album_tracks = df[at_cols].drop_duplicates()
    at_rename = {"album_id_norm": "album_id", "track_id_norm": "track_id"}
    if c_disc_number:
        at_rename[c_disc_number] = "disc_number"
    if c_track_number:
        at_rename[c_track_number] = "track_number"
    album_tracks = album_tracks.rename(columns=at_rename)

    if "disc_number" not in album_tracks.columns:
        album_tracks["disc_number"] = 1
    if "track_number" not in album_tracks.columns:
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
    # TRACK_GENRES (one primary Big-8 per track: first genre in CSV for that track; used for song/album primary genre)
    # -----------------------
    track_genres_rows = []
    if c_genres:
        track_to_genres = df.set_index("track_id_norm")[c_genres].to_dict()
        for tid in df["track_id_norm"].drop_duplicates().tolist():
            gval = track_to_genres.get(tid, None)
            parts = split_multi(gval)
            if not parts:
                continue
            first_genre = re.sub(r"\s+", " ", parts[0].strip().lower())[:120]
            if not first_genre:
                continue
            g_group = map_genre_to_group(first_genre)
            if g_group:
                track_genres_rows.append((tid, g_group))
    track_genres = (
        pd.DataFrame(track_genres_rows, columns=["track_id", "genre_name"]).drop_duplicates(subset=["track_id"], keep="first")
        if track_genres_rows else pd.DataFrame(columns=["track_id", "genre_name"])
    )

    # -----------------------
    # ARTIST_PRIMARY_GENRE (one Big-8 per artist: primary genre of that artist's highest-popularity track that has track_genres; aligned with song/album)
    # -----------------------
    if len(track_genres) > 0 and len(track_artist) > 0 and "popularity" in tracks.columns:
        apg = track_artist.merge(track_genres, on="track_id").merge(
            tracks[["track_id", "popularity"]], on="track_id"
        )
        apg = apg.sort_values(["artist_id", "popularity"], ascending=[True, False]).drop_duplicates(
            subset=["artist_id"], keep="first"
        )
        artist_primary_genre = apg[["artist_id", "genre_name"]].copy()
    else:
        artist_primary_genre = pd.DataFrame(columns=["artist_id", "genre_name"])

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

    if do_print_columns:
        print("=== Column names after ETL ===\n")
        print("[release_date mapping]")
        print("  Source column: 'Album Release Date' (CSV with spaces)")
        print("  -> Renamed: df['album_release_date']")
        print("  -> find_col('release_date') yields: c_release_date =", repr(c_release_date))
        print("  -> albums table column: 'release_date'")
        print("  -> DB: albums.release_date -> DB albums.release_date")
        non_null = albums["release_date"].notna().sum() if "release_date" in albums.columns else 0
        print("  Sample: albums.release_date non-null count =", non_null, "/", len(albums))
        if "release_date" in albums.columns and non_null > 0:
            sample = albums["release_date"].dropna().head(5).tolist()
            print("  First 5 non-null values:", sample)
        print()
        print("df (after CSV processing):", list(df.columns))
        print("artists:", list(artists.columns))
        print("albums:", list(albums.columns))
        print("tracks:", list(tracks.columns))
        print("track_artist:", list(track_artist.columns))
        print("album_tracks:", list(album_tracks.columns))
        print("artist_genres:", list(artist_genres.columns))
        print("track_genres:", list(track_genres.columns))
        print("artist_primary_genre:", list(artist_primary_genre.columns))
        print("audio_features:", list(audio_features.columns))
        return

    # -----------------------
    # LOAD TO POSTGRES
    # -----------------------
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        if albums_only:
            cur.execute("TRUNCATE album_tracks")
            cur.execute("TRUNCATE albums CASCADE")
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
                "INSERT INTO albums (album_id, album_name, release_date, album_image_url, status) VALUES %s ",
                album_rows,
                page_size=2000
            )
            execute_values(
                cur,
                "INSERT INTO album_tracks (album_id, track_id, disc_number, track_number) VALUES %s ",
                list(album_tracks.itertuples(index=False, name=None)),
                page_size=5000
            )
            conn.commit()
            print("✅ Albums-only reload completed.")
            print(f"Albums: {len(albums)} | Album_Tracks: {len(album_tracks)}")
            cur.close()
            conn.close()
            return

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
            "ON CONFLICT (album_id) DO UPDATE SET album_name=EXCLUDED.album_name, release_date=EXCLUDED.release_date, album_image_url=EXCLUDED.album_image_url",
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
            added_raw = r.get("added_at")
            if added_raw is None or pd.isna(added_raw):
                added_val = None
            else:
                s = str(added_raw).strip()
                # handle possible ISO datetime like 2024-08-04T...
                if "T" in s:
                    s = s.split("T", 1)[0]
                added_val = s or None
            track_rows.append((r["track_id"], r["track_name"], pop, dur, exp, prv, "approved", added_val))

        execute_values(
            cur,
            "INSERT INTO tracks (track_id, track_name, popularity, duration_ms, explicit, preview_url, status, added_at) VALUES %s "
            "ON CONFLICT (track_id) DO UPDATE SET track_name=EXCLUDED.track_name, added_at=COALESCE(EXCLUDED.added_at, tracks.added_at)",
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

        # artist_genres: truncate then insert to avoid stale mapping (e.g. previously misclassified Electronic)
        cur.execute("TRUNCATE TABLE artist_genres")
        if len(artist_genres) > 0:
            execute_values(
                cur,
                "INSERT INTO artist_genres (artist_id, genre_name) VALUES %s ON CONFLICT DO NOTHING",
                list(artist_genres.itertuples(index=False, name=None)),
                page_size=5000
            )

        # artist_primary_genre: one primary genre per artist from highest-popularity track (aligned with song/album)
        cur.execute(
            "CREATE TABLE IF NOT EXISTS artist_primary_genre (artist_id VARCHAR(64) PRIMARY KEY REFERENCES artists(artist_id) ON DELETE CASCADE, genre_name VARCHAR(120) NOT NULL)"
        )
        cur.execute("TRUNCATE TABLE artist_primary_genre")
        if len(artist_primary_genre) > 0:
            execute_values(
                cur,
                "INSERT INTO artist_primary_genre (artist_id, genre_name) VALUES %s ON CONFLICT (artist_id) DO UPDATE SET genre_name = EXCLUDED.genre_name",
                list(artist_primary_genre.itertuples(index=False, name=None)),
                page_size=5000
            )

        # track_genres: create table if needed, truncate, then insert; one primary Big-8 per track from first genre in CSV
        if len(track_genres) > 0:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS track_genres (track_id VARCHAR(64) PRIMARY KEY REFERENCES tracks(track_id) ON DELETE CASCADE, genre_name VARCHAR(120) NOT NULL)"
            )
            cur.execute("TRUNCATE TABLE track_genres")
            execute_values(
                cur,
                "INSERT INTO track_genres (track_id, genre_name) VALUES %s ON CONFLICT (track_id) DO UPDATE SET genre_name = EXCLUDED.genre_name",
                list(track_genres.itertuples(index=False, name=None)),
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
        print(f"Track_Artist: {len(track_artist)} | Album_Tracks: {len(album_tracks)} | Artist_Genres(rows): {len(artist_genres)} | Artist_Primary_Genre: {len(artist_primary_genre)} | Track_Genres: {len(track_genres)}")
        print(f"Audio_Features: {len(audio_features) if have_audio else 0}")

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
