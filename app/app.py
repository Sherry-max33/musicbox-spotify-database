# app/app.py
"""
Viewer sorting/ranking logic for /viewer (Song / Artist / Album Top Chart):

- Song Top Chart
  - Filter: t.status = 'approved', with optional decade filter by album release_date.
  - Dedupe: DISTINCT ON (t.track_id); one row per track (when multiple albums/artists, keep row with highest popularity).
  - Sort: popularity DESC NULLS LAST, top 50 (frontend shows top 10).

- Artist Top Chart
  - Filter: same as Song; decade filter by album release_date.
  - Score: same track in multiple albums counts multiple times (no dedupe); score = SUM(popularity) per artist.
  - Per-artist TOP TRACK: that artist's track with highest popularity in the filtered set (ROW_NUMBER() rn=1).
  - Sort: score DESC NULLS LAST, top 10.

- Album Top Chart
  - Filter: same as Song; t.status = 'approved', decade by album release_date.
  - Score: album score = sum of popularity of all tracks in album (each track counted once; SUM(t.popularity) per album).
  - Album artist: artist with highest total popularity in that album (avoids MIN(artist_name) showing feat. artist by alphabet).
  - Sort: album score DESC NULLS LAST, top 10.

- EXPLORE BY GENRES (lower part of /viewer)
  - Songs/albums/artists filtered by "primary" genre, aligned with ETL: songs/albums use track_genres (first genre in CSV mapped to Big-8); artists use artist_primary_genre (primary genre of that artist's highest-popularity track that has track_genres).
  - API: GET /viewer/api/genre_chart?genre=Pop&decade=all&type=songs|artists|albums; same logic as above; genre filter: track_genres for songs/albums, artist_primary_genre for artists; returns top 5 only.
  - Frontend keeps current layout; requests API by selected genre, decade, and Songs/Artists/Albums and renders grid.

- Artist page /artist?artist_id=xxx
  - Same as Viewer Artist: total popularity = SUM(popularity) (all tracks for that artist; same track in multiple albums counted multiple times). Rank = full DB artists ordered by that SUM descending.
  - Top-right: first line shows # {rank}, second line Popularity {total_popularity}. TOP TRACKS and ALBUMS are queried and rendered by backend.
"""
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import psycopg
from dotenv import load_dotenv
import requests

try:
    import cv2
    import numpy as np
    _FACE_DETECT_AVAILABLE = True
except Exception:
    cv2 = np = None
    _FACE_DETECT_AVAILABLE = False

load_dotenv()

# =========================
# DB connection
# =========================
def get_conn():
    # Prefer a single DATABASE_URL if present (Render / Supabase style),
    # otherwise fall back to individual DB_* pieces (local / custom).
    url = os.getenv("DATABASE_URL")
    if url:
        return psycopg.connect(url)
    return psycopg.connect(
        host=os.getenv("DB_HOST") or "localhost",
        port=int(os.getenv("DB_PORT") or 5432),
        dbname=os.getenv("DB_NAME") or "musicbox",
        user=os.getenv("DB_USER") or os.getenv("USER"),
        password=os.getenv("DB_PASSWORD") or None,
        sslmode=os.getenv("DB_SSLMODE") or None,
    )

app = Flask(__name__)
app.secret_key = "dev-secret"

# Big-8 categories (same as ETL GENRE_GROUPS; includes Country)
BIG7 = ["Pop", "Rock", "Hip-Hop", "R&B", "Jazz", "Classical", "Electronic", "Country"]

# Tabs for Trend chart (mock)
FEATURE_TABS = ["danceability", "energy", "valence",
                "acousticness", "tempo", "loudness", "duration"]


# =========================
# Helpers
# =========================
def decade_label(decade_int: int) -> str:
    return f"{int(decade_int)}s"


def fetchone_dict(cur, columns):
    row = cur.fetchone()
    if not row:
        return None
    return {columns[i]: row[i] for i in range(len(columns))}


# Artist page hero background: compute background-position from face position in album art (cached by URL)
_hero_focus_cache = {}


def get_face_focus(image_url, timeout=4):
    """Detect face in album image and return CSS background-position percentages so the face stays in view. Returns None if no face or on failure."""
    if not _FACE_DETECT_AVAILABLE or not image_url or not image_url.startswith("http"):
        return None
    if image_url in _hero_focus_cache:
        return _hero_focus_cache[image_url]
    try:
        r = requests.get(image_url, timeout=timeout)
        r.raise_for_status()
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            _hero_focus_cache[image_url] = None
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            _hero_focus_cache[image_url] = None
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        cx = (x + w / 2) / img.shape[1]
        cy = (y + h / 2) / img.shape[0]
        # Map face center to viewport relative position for consistent cropping
        out = f"{cx*100:.1f}% {cy*100:.1f}%"
        _hero_focus_cache[image_url] = out
        return out
    except Exception:
        _hero_focus_cache[image_url] = None
        return None


# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("viewer"))


def _search_all(q, limit=8):
    """Return { artists, albums, tracks } each list of { id, name } for dropdown / results page."""
    if not q or not q.strip():
        return {"artists": [], "albums": [], "tracks": []}
    q = q.strip()[:120]
    like = f"%{q}%"
    out = {"artists": [], "albums": [], "tracks": []}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT artist_id, artist_name FROM artists
                   WHERE status = 'approved' AND artist_name ILIKE %s
                   ORDER BY artist_name LIMIT %s""",
                (like, limit),
            )
            out["artists"] = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
            cur.execute(
                """SELECT album_id, album_name FROM albums
                   WHERE status = 'approved' AND album_name ILIKE %s
                   ORDER BY album_name LIMIT %s""",
                (like, limit),
            )
            out["albums"] = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
            cur.execute(
                """SELECT track_id, track_name FROM tracks
                   WHERE status = 'approved' AND track_name ILIKE %s
                   ORDER BY popularity DESC NULLS LAST LIMIT %s""",
                (like, limit),
            )
            out["tracks"] = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
    return out


@app.route("/search/api", methods=["GET"])
def search_api():
    """JSON: { artists: [{id,name}], albums: [{id,name}], tracks: [{id,name}] } for dropdown."""
    q = request.args.get("q", "").strip()
    data = _search_all(q, limit=8)
    return jsonify(data)


@app.route("/search", methods=["GET"])
def search():
    """Search results page: show grouped matches (artists / albums / tracks) with links."""
    q = request.args.get("q", "").strip()
    if not q:
        return redirect(url_for("viewer"))
    data = _search_all(q, limit=10)
    return render_template(
        "search_results.html",
        q=q,
        artists=data["artists"],
        albums=data["albums"],
        tracks=data["tracks"],
    )


# -------------------------
# VIEWER (dynamic: query by decade / genre / search)
# -------------------------
@app.route("/viewer", methods=["GET"])
def viewer():
    chart = request.args.get("chart", "song").strip().lower()
    if chart not in ("song", "artist", "album"):
        chart = "song"
    decade = request.args.get("decade", "All")
    genre = request.args.get("genre", "")
    q = request.args.get("q", "").strip()

    decade_start = None
    decade_end = None
    if decade and decade != "All":
        try:
            start_year = int(str(decade).rstrip("s"))
            decade_start = start_year
            decade_end = start_year + 9
        except (ValueError, AttributeError):
            pass
    has_decade_filter = decade_start is not None and decade_end is not None

    if chart == "song":
        # Song chart: each track shows all associated artists (not just one)
        decade_condition = (
            """
            AND al.release_date IS NOT NULL
              AND EXTRACT(YEAR FROM al.release_date)
                  BETWEEN %(decade_start)s AND %(decade_end)s
            """
            if has_decade_filter
            else ""
        )
        song_sql = f"""
        WITH decade_tracks AS (
          SELECT
            t.track_id,
            t.track_name,
            t.popularity,
            t.preview_url,
            ar.artist_id,
            ar.artist_name,
            al.album_name,
            al.album_image_url,
            al.release_date
          FROM tracks t
          JOIN track_artist ta ON ta.track_id = t.track_id
          JOIN artists ar ON ar.artist_id = ta.artist_id
          JOIN album_tracks at ON at.track_id = t.track_id
          JOIN albums al ON al.album_id = at.album_id
          WHERE t.status = 'approved'
{decade_condition}
        ),
        dedup AS (
          SELECT DISTINCT ON (track_id)
            track_id, track_name, popularity, preview_url,
            album_name, album_image_url, release_date
          FROM decade_tracks
          ORDER BY track_id, popularity DESC NULLS LAST
        ),
        artist_agg AS (
          SELECT
            track_id,
            array_agg(artist_id ORDER BY artist_name) AS artist_ids,
            array_agg(artist_name ORDER BY artist_name) AS artist_names
          FROM decade_tracks
          GROUP BY track_id
        )
        SELECT
          d.track_id,
          d.track_name,
          d.popularity,
          d.preview_url,
          a.artist_ids,
          a.artist_names,
          d.album_name,
          d.album_image_url,
          d.release_date
        FROM dedup d
        JOIN artist_agg a USING (track_id)
        ORDER BY d.popularity DESC NULLS LAST
        LIMIT 50;
        """
        with get_conn() as conn:
            params = {}
            if has_decade_filter:
                params = {"decade_start": decade_start, "decade_end": decade_end}
            with conn.cursor() as cur:
                cur.execute(song_sql, params)
                rows = cur.fetchall()

    elif chart == "artist":
        # Same as Song Top Chart: decade filter by album release_date; same track in multiple albums counts multiple times (no dedupe)
        decade_condition = (
            """
            AND al.release_date IS NOT NULL
              AND EXTRACT(YEAR FROM al.release_date)
                  BETWEEN %(decade_start)s AND %(decade_end)s
            """
            if has_decade_filter
            else ""
        )
        artist_sql = f"""
        WITH decade_tracks AS (
          SELECT t.track_id, t.track_name, t.preview_url, t.popularity, ar.artist_id, ar.artist_name
          FROM tracks t
          JOIN track_artist ta ON ta.track_id = t.track_id
          JOIN artists ar ON ar.artist_id = ta.artist_id
          JOIN album_tracks at ON at.track_id = t.track_id
          JOIN albums al ON al.album_id = at.album_id
          WHERE t.status = 'approved'
{decade_condition}
        ),
        ranked AS (
          SELECT artist_id,
                 artist_name,
                 track_id,
                 track_name,
                 preview_url,
                 popularity,
                 ROW_NUMBER() OVER (PARTITION BY artist_id ORDER BY popularity DESC NULLS LAST) AS rn
          FROM decade_tracks
        ),
        artist_score AS (
          SELECT artist_id, SUM(popularity)::numeric AS score FROM decade_tracks GROUP BY artist_id
        )
        SELECT r.artist_name, r.track_name, r.preview_url, r.artist_id, r.track_id
        FROM ranked r
        JOIN artist_score s ON s.artist_id = r.artist_id
        WHERE r.rn = 1
        ORDER BY s.score DESC NULLS LAST
        LIMIT 10;
        """
        with get_conn() as conn:
            params = {}
            if has_decade_filter:
                params = {"decade_start": decade_start, "decade_end": decade_end}
            with conn.cursor() as cur:
                cur.execute(artist_sql, params)
                rows = cur.fetchall()

    elif chart == "album":
        # Album score = sum of track popularity in album (each track once); album artist = artist with highest total popularity in that album
        decade_condition_a = (
            """
            AND a.release_date IS NOT NULL
              AND EXTRACT(YEAR FROM a.release_date)
                  BETWEEN %(decade_start)s AND %(decade_end)s
            """
            if has_decade_filter
            else ""
        )
        album_sql = f"""
        WITH album_total AS (
          SELECT a.album_id, a.album_name, a.album_image_url,
                 SUM(t.popularity) AS album_total
          FROM albums a
          JOIN album_tracks at ON at.album_id = a.album_id
          JOIN tracks t ON t.track_id = at.track_id
          WHERE t.status = 'approved'
{decade_condition_a}
          GROUP BY a.album_id, a.album_name, a.album_image_url
        ),
        album_artist_pop AS (
          SELECT a.album_id, ar.artist_id, ar.artist_name, SUM(t.popularity) AS artist_pop
          FROM albums a
          JOIN album_tracks at ON at.album_id = a.album_id
          JOIN tracks t ON t.track_id = at.track_id
          JOIN track_artist ta ON ta.track_id = t.track_id
          JOIN artists ar ON ar.artist_id = ta.artist_id
          WHERE t.status = 'approved'
{decade_condition_a}
          GROUP BY a.album_id, ar.artist_id, ar.artist_name
        ),
        best_artist AS (
          SELECT DISTINCT ON (album_id) album_id, artist_id, artist_name
          FROM album_artist_pop
          ORDER BY album_id, artist_pop DESC NULLS LAST
        )
        SELECT tot.album_id, tot.album_name, tot.album_image_url, ba.artist_id, ba.artist_name
        FROM album_total tot
        JOIN best_artist ba ON ba.album_id = tot.album_id
        ORDER BY tot.album_total DESC NULLS LAST
        LIMIT 10;
        """
        with get_conn() as conn:
            params = {}
            if has_decade_filter:
                params = {"decade_start": decade_start, "decade_end": decade_end}
            with conn.cursor() as cur:
                cur.execute(album_sql, params)
                rows = cur.fetchall()

    else:
        chart = "song"
        where = ["t.status = 'approved'"]
        params = []
        if q:
            where.append("(lower(t.track_name) LIKE lower(%s) OR lower(ar.artist_name) LIKE lower(%s) OR lower(a.album_name) LIKE lower(%s))")
            params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
        sql = f"""
        SELECT t.track_id, t.track_name, ar.artist_name, a.album_name, t.popularity, a.album_image_url, t.preview_url
        FROM tracks t
        JOIN album_tracks at ON at.track_id = t.track_id
        JOIN albums a ON a.album_id = at.album_id
        JOIN track_artist ta ON ta.track_id = t.track_id
        JOIN artists ar ON ar.artist_id = ta.artist_id
        WHERE {" AND ".join(where)}
        ORDER BY t.popularity DESC NULLS LAST
        LIMIT 50;
        """
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

    decades = ["All", "1980s", "1990s", "2000s", "2010s", "2020s"]
    genres = list(BIG7)

    return render_template(
        "viewer.html",
        rows=rows,
        genres=genres,
        decades=decades,
        decade=decade,
        genre=genre,
        q=q,
        chart=chart,
    )


# -------------------------
# Genre chart API (EXPLORE BY GENRES): same logic as top chart, filter by genre, top 5
# -------------------------
def _decade_to_range(decade_val):
    """'All' or None -> (None, None); '1980s' -> (1980, 1989)."""
    if not decade_val or str(decade_val).strip().lower() == "all":
        return None, None
    try:
        start = int(str(decade_val).rstrip("s"))
        return start, start + 9
    except (ValueError, AttributeError):
        return None, None


@app.route("/viewer/api/genre_chart", methods=["GET"])
def viewer_genre_chart():
    genre = request.args.get("genre", "").strip()
    if genre not in BIG7:
        return jsonify({"error": "invalid genre", "items": []}), 400
    decade_param = request.args.get("decade", "all").strip() or "all"
    chart_type = request.args.get("type", "songs").strip().lower()
    if chart_type not in ("songs", "artists", "albums"):
        chart_type = "songs"
    decade_start, decade_end = _decade_to_range(decade_param)
    has_decade_filter = decade_start is not None and decade_end is not None
    base_params = {"genre": genre}
    items = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            if chart_type == "songs":
                decade_condition = (
                    """
                    AND al.release_date IS NOT NULL
                      AND EXTRACT(YEAR FROM al.release_date)
                          BETWEEN %(decade_start)s AND %(decade_end)s
                    """
                    if has_decade_filter
                    else ""
                )
                sql = f"""
                WITH genre_tracks AS (
                  SELECT
                    t.track_id, t.track_name, t.popularity, t.preview_url,
                    ar.artist_id, ar.artist_name,
                    al.album_id, al.album_name, al.album_image_url
                  FROM tracks t
                  JOIN track_genres tg ON tg.track_id = t.track_id AND tg.genre_name = %(genre)s
                  JOIN track_artist ta ON ta.track_id = t.track_id
                  JOIN artists ar ON ar.artist_id = ta.artist_id
                  JOIN album_tracks at ON at.track_id = t.track_id
                  JOIN albums al ON al.album_id = at.album_id
                  WHERE t.status = 'approved'
{decade_condition}
                ),
                dedup AS (
                  SELECT DISTINCT ON (track_id)
                    track_id, track_name, popularity, preview_url,
                    album_id, album_name, album_image_url
                  FROM genre_tracks
                  ORDER BY track_id, popularity DESC NULLS LAST
                ),
                artist_agg AS (
                  SELECT
                    track_id,
                    array_agg(artist_id ORDER BY artist_name) AS artist_ids,
                    array_agg(artist_name ORDER BY artist_name) AS artist_names
                  FROM genre_tracks
                  GROUP BY track_id
                )
                SELECT
                  d.track_id,
                  d.track_name,
                  d.popularity,
                  d.preview_url,
                  a.artist_ids,
                  a.artist_names,
                  d.album_id,
                  d.album_name,
                  d.album_image_url
                FROM dedup d
                JOIN artist_agg a USING (track_id)
                ORDER BY d.popularity DESC NULLS LAST
                LIMIT 5;
                """
                params = dict(base_params)
                if has_decade_filter:
                    params.update({"decade_start": decade_start, "decade_end": decade_end})
                cur.execute(sql, params)
                rows = cur.fetchall()
                for i, r in enumerate(rows):
                    artist_ids = r[4] or []
                    artist_names = r[5] or []
                    primary_artist_id = artist_ids[0] if artist_ids else None
                    display_name = ", ".join(artist_names) if artist_names else None
                    items.append(
                        {
                            "rank": i + 1,
                            "track_id": r[0],
                            "track_name": r[1],
                            "artist_id": primary_artist_id,
                            "artist_name": display_name,
                            "album_id": r[6],
                            "album_name": r[7],
                            "album_image_url": r[8],
                            "preview_url": r[3],
                        }
                    )
            elif chart_type == "artists":
                decade_condition = (
                    """
                    AND al.release_date IS NOT NULL
                      AND EXTRACT(YEAR FROM al.release_date)
                          BETWEEN %(decade_start)s AND %(decade_end)s
                    """
                    if has_decade_filter
                    else ""
                )
                sql = f"""
                WITH decade_tracks AS (
                  SELECT t.track_id, t.track_name, t.preview_url, t.popularity, ar.artist_id, ar.artist_name, al.album_image_url
                  FROM tracks t
                  JOIN track_artist ta ON ta.track_id = t.track_id
                  JOIN artists ar ON ar.artist_id = ta.artist_id
                  JOIN artist_primary_genre apg ON apg.artist_id = ar.artist_id AND apg.genre_name = %(genre)s
                  JOIN album_tracks at ON at.track_id = t.track_id
                  JOIN albums al ON al.album_id = at.album_id
                  WHERE t.status = 'approved'
{decade_condition}
                ),
                ranked AS (
                  SELECT artist_id, artist_name, track_name, preview_url, album_image_url,
                         ROW_NUMBER() OVER (PARTITION BY artist_id ORDER BY popularity DESC NULLS LAST) AS rn
                  FROM decade_tracks
                ),
                artist_score AS (
                  SELECT artist_id, SUM(popularity)::numeric AS score FROM decade_tracks GROUP BY artist_id
                ),
                one_per_artist AS (
                  SELECT DISTINCT ON (artist_id) artist_id, artist_name, track_name, preview_url, album_image_url
                  FROM ranked WHERE rn = 1 ORDER BY artist_id
                )
                SELECT o.artist_id, o.artist_name, o.track_name, o.preview_url, o.album_image_url
                FROM one_per_artist o
                JOIN artist_score s ON s.artist_id = o.artist_id
                ORDER BY s.score DESC NULLS LAST
                LIMIT 5;
                """
                params = dict(base_params)
                if has_decade_filter:
                    params.update({"decade_start": decade_start, "decade_end": decade_end})
                cur.execute(sql, params)
                rows = cur.fetchall()
                items = [
                    {
                        "rank": i + 1,
                        "artist_id": r[0],
                        "artist_name": r[1],
                        "top_track_name": r[2],
                        "preview_url": r[3],
                        "album_image_url": r[4],
                    }
                    for i, r in enumerate(rows)
                ]
            else:
                decade_condition = (
                    """
                    AND a.release_date IS NOT NULL
                      AND EXTRACT(YEAR FROM a.release_date)
                          BETWEEN %(decade_start)s AND %(decade_end)s
                    """
                    if has_decade_filter
                    else ""
                )
                sql = f"""
                WITH album_total AS (
                  SELECT a.album_id, a.album_name, a.album_image_url,
                         SUM(t.popularity) AS album_total
                  FROM albums a
                  JOIN album_tracks at ON at.album_id = a.album_id
                  JOIN tracks t ON t.track_id = at.track_id
                  WHERE t.status = 'approved'
                    {decade_condition}
                    AND EXISTS (
                      SELECT 1 FROM album_tracks at2
                      JOIN track_genres tg ON tg.track_id = at2.track_id AND tg.genre_name = %(genre)s
                      WHERE at2.album_id = a.album_id
                    )
                  GROUP BY a.album_id, a.album_name, a.album_image_url
                ),
                album_artist_pop AS (
                  SELECT a.album_id, ar.artist_id, ar.artist_name, SUM(t.popularity) AS artist_pop
                  FROM albums a
                  JOIN album_tracks at ON at.album_id = a.album_id
                  JOIN tracks t ON t.track_id = at.track_id
                  JOIN track_artist ta ON ta.track_id = t.track_id
                  JOIN artists ar ON ar.artist_id = ta.artist_id
                  WHERE t.status = 'approved'
                    {decade_condition}
                  GROUP BY a.album_id, ar.artist_id, ar.artist_name
                ),
                best_artist AS (
                  SELECT DISTINCT ON (album_id) album_id, artist_id, artist_name
                  FROM album_artist_pop
                  ORDER BY album_id, artist_pop DESC NULLS LAST
                )
                SELECT tot.album_id, tot.album_name, tot.album_image_url, ba.artist_id, ba.artist_name
                FROM album_total tot
                JOIN best_artist ba ON ba.album_id = tot.album_id
                ORDER BY tot.album_total DESC NULLS LAST
                LIMIT 5;
                """
                params = dict(base_params)
                if has_decade_filter:
                    params.update({"decade_start": decade_start, "decade_end": decade_end})
                cur.execute(sql, params)
                rows = cur.fetchall()
                items = [
                    {
                        "rank": i + 1,
                        "album_id": r[0],
                        "album_name": r[1],
                        "album_image_url": r[2],
                        "artist_id": r[3],
                        "artist_name": r[4],
                    }
                    for i, r in enumerate(rows)
                ]
    return jsonify({"items": items, "genre": genre, "decade": decade_param, "type": chart_type})


@app.route("/album", methods=["GET"])
def album():
    album_id = request.args.get("album_id", "").strip()
    if not album_id:
        return render_template(
            "album.html",
            album_id=None,
            album_name=None,
            album_image_url=None,
            release_year=None,
            total_popularity=None,
            artist_id=None,
            artist_name=None,
            tracks=[],
        )

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Basic info + total popularity + primary artist (artist with highest total popularity in album)
            cur.execute(
                """
                WITH album_tracks AS (
                  SELECT t.track_id,
                         t.track_name,
                         t.popularity,
                         t.preview_url,
                         at.track_number,
                         ar.artist_id,
                         ar.artist_name
                  FROM albums a
                  JOIN album_tracks at ON at.album_id = a.album_id
                  JOIN tracks t ON t.track_id = at.track_id
                  LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                  LEFT JOIN artists ar ON ar.artist_id = ta.artist_id
                  WHERE a.album_id = %(album_id)s
                    AND t.status = 'approved'
                ),
                album_total AS (
                  SELECT COALESCE(SUM(popularity), 0)::int AS total_pop FROM album_tracks
                ),
                artist_score AS (
                  SELECT artist_id, artist_name, SUM(popularity) AS artist_pop
                  FROM album_tracks
                  WHERE artist_id IS NOT NULL
                  GROUP BY artist_id, artist_name
                ),
                best_artist AS (
                  SELECT artist_id, artist_name
                  FROM artist_score
                  ORDER BY artist_pop DESC NULLS LAST
                  LIMIT 1
                )
                SELECT
                  a.album_name,
                  a.album_image_url,
                  a.release_date,
                  (SELECT total_pop FROM album_total),
                  (SELECT artist_id FROM best_artist),
                  (SELECT artist_name FROM best_artist)
                FROM albums a
                WHERE a.album_id = %(album_id)s;
                """,
                {"album_id": album_id},
            )
            row = cur.fetchone()
            if not row:
                return render_template(
                    "album.html",
                    album_id=album_id,
                    album_name=None,
                    album_image_url=None,
                    release_year=None,
                    total_popularity=None,
                    artist_id=None,
                    artist_name=None,
                    tracks=[],
                )

            album_name, album_image_url, release_date, total_popularity, artist_id, artist_name = row
            release_year = str(release_date)[:4] if release_date else None

            # All tracks in album (ordered by track_number; unnumbered last, then by track name)
            cur.execute(
                """
                WITH base AS (
                  SELECT
                    t.track_id,
                    t.track_name,
                    t.popularity,
                    t.preview_url,
                    at.track_number
                  FROM albums a
                  JOIN album_tracks at ON at.album_id = a.album_id
                  JOIN tracks t ON t.track_id = at.track_id
                  WHERE a.album_id = %(album_id)s
                    AND t.status = 'approved'
                ),
                artists_per_track AS (
                  SELECT
                    ta.track_id,
                    array_agg(ar.artist_id ORDER BY ar.artist_name) AS artist_ids,
                    array_agg(ar.artist_name ORDER BY ar.artist_name) AS artist_names
                  FROM track_artist ta
                  JOIN artists ar ON ar.artist_id = ta.artist_id
                  WHERE ta.track_id IN (SELECT track_id FROM base)
                  GROUP BY ta.track_id
                )
                SELECT
                  b.track_id,
                  b.track_name,
                  COALESCE(ap.artist_ids, ARRAY[]::text[]) AS artist_ids,
                  COALESCE(ap.artist_names, ARRAY[]::text[]) AS artist_names,
                  b.popularity,
                  b.preview_url,
                  b.track_number
                FROM base b
                LEFT JOIN artists_per_track ap ON ap.track_id = b.track_id
                ORDER BY b.track_number NULLS LAST, b.track_name;
                """,
                {"album_id": album_id},
            )
            track_rows = cur.fetchall()
            tracks = [
                {
                    "track_id": r[0],
                    "track_name": r[1],
                    "artists": [
                        {"artist_id": aid, "artist_name": aname}
                        for (aid, aname) in zip((r[2] or []), (r[3] or []))
                    ],
                    "artist_names": ", ".join((r[3] or [])),
                    "popularity": r[4],
                    "preview_url": r[5],
                    "track_number": r[6],
                }
                for r in track_rows
            ]

    return render_template(
        "album.html",
        album_id=album_id,
        album_name=album_name,
        album_image_url=album_image_url,
        release_year=release_year,
        total_popularity=total_popularity,
        artist_id=artist_id,
        artist_name=artist_name,
        tracks=tracks,
    )


@app.route("/track", methods=["GET"])
def track():
    track_id = request.args.get("track_id", "").strip()
    if not track_id:
        return render_template(
            "track.html",
            track_id=None,
            track_name=None,
            artists=[],
            album_id=None,
            album_name=None,
            album_image_url=None,
            release_year=None,
            duration_ms=None,
            duration_str=None,
            popularity=None,
            explicit=False,
            preview_url=None,
        )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH base AS (
                  SELECT
                    t.track_id,
                    t.track_name,
                    t.duration_ms,
                    t.explicit,
                    t.preview_url,
                    t.popularity,
                    a.album_id,
                    a.album_name,
                    a.album_image_url,
                    a.release_date
                  FROM tracks t
                  JOIN album_tracks at ON at.track_id = t.track_id
                  JOIN albums a ON a.album_id = at.album_id
                  WHERE t.track_id = %(track_id)s
                    AND t.status = 'approved'
                  ORDER BY COALESCE(at.disc_number, 1), COALESCE(at.track_number, 1)
                  LIMIT 1
                ),
                artists AS (
                  SELECT
                    ta.track_id,
                    array_agg(ar.artist_id ORDER BY ar.artist_name) AS artist_ids,
                    array_agg(ar.artist_name ORDER BY ar.artist_name) AS artist_names
                  FROM track_artist ta
                  JOIN artists ar ON ar.artist_id = ta.artist_id
                  WHERE ta.track_id = %(track_id)s
                  GROUP BY ta.track_id
                )
                SELECT
                  b.track_name,
                  b.duration_ms,
                  b.explicit,
                  b.preview_url,
                  b.popularity,
                  b.album_id,
                  b.album_name,
                  b.album_image_url,
                  b.release_date,
                  a.artist_ids,
                  a.artist_names
                FROM base b
                LEFT JOIN artists a ON a.track_id = b.track_id;
                """,
                {"track_id": track_id},
            )
            row = cur.fetchone()
            if not row:
                return render_template(
                    "track.html",
                    track_id=track_id,
                    track_name=None,
                    artists=[],
                    album_id=None,
                    album_name=None,
                    album_image_url=None,
                    release_year=None,
                    duration_ms=None,
                    duration_str=None,
                    popularity=None,
                    explicit=False,
                    preview_url=None,
                )

            (
                track_name,
                duration_ms,
                explicit,
                preview_url,
                popularity,
                album_id,
                album_name,
                album_image_url,
                release_date,
                artist_ids,
                artist_names,
            ) = row

            release_year = str(release_date)[:4] if release_date else None
            if duration_ms is not None:
                total_seconds = int(duration_ms) // 1000
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                duration_str = f"{minutes}:{seconds:02d}"
            else:
                duration_str = None

            artists = []
            if artist_ids and artist_names:
                for i, name in enumerate(artist_names):
                    artists.append({"id": artist_ids[i], "name": name})

    return render_template(
        "track.html",
        track_id=track_id,
        track_name=track_name,
        artists=artists,
        album_id=album_id,
        album_name=album_name,
        album_image_url=album_image_url,
        release_year=release_year,
        duration_ms=duration_ms,
        duration_str=duration_str,
        popularity=popularity,
        explicit=explicit,
        preview_url=preview_url,
    )


# -------------------------
# Analyst (UNCHANGED)
# -------------------------
@app.route("/analyst", methods=["GET"])
def analyst():
    genre = request.args.get("genre", "").strip()
    trend_genre = request.args.get("trend_genre", genre).strip()
    feature = request.args.get("feature", "danceability").strip()

    if feature not in FEATURE_TABS:
        feature = "danceability"

    if genre and genre not in BIG7:
        genre = ""
    if trend_genre and trend_genre not in BIG7:
        trend_genre = genre

    genres_sql = """
    SELECT genre_name
    FROM (
      SELECT DISTINCT genre_name FROM artist_genres
    ) x
    WHERE genre_name = ANY(%s)
    ORDER BY genre_name;
    """

    base_cte = f"""
    WITH base AS (
      SELECT DISTINCT
        t.track_id,
        (FLOOR(EXTRACT(YEAR FROM a.release_date)/10)*10)::int AS decade,
        t.duration_ms,
        af.danceability,
        af.energy,
        af.valence,
        af.acousticness,
        af.tempo,
        af.loudness,
        (
          SELECT MIN(ag2.genre_name)
          FROM track_artist ta2
          JOIN artist_genres ag2 ON ag2.artist_id = ta2.artist_id
          WHERE ta2.track_id = t.track_id
            AND ag2.genre_name = ANY(%s)
        ) AS genre_group
      FROM tracks t
      JOIN audio_features af ON af.track_id = t.track_id
      JOIN album_tracks at ON at.track_id = t.track_id
      JOIN albums a ON a.album_id = at.album_id
      WHERE t.status='approved'
        AND a.release_date IS NOT NULL
    )
    """

    summary_sql = base_cte + """
    SELECT
      ROUND(AVG(danceability)::numeric, 3) AS danceability,
      ROUND(AVG(energy)::numeric, 3)       AS energy,
      ROUND(AVG(valence)::numeric, 3)      AS valence,
      ROUND(AVG(acousticness)::numeric, 3) AS acousticness,
      ROUND(AVG(tempo)::numeric, 2)        AS tempo,
      ROUND(AVG(loudness)::numeric, 2)     AS loudness,
      ROUND(AVG(duration_ms)::numeric, 0)  AS duration_ms
    FROM base
    WHERE genre_group IS NOT NULL
      AND (%s = '' OR genre_group = %s);
    """

    mix_sql = base_cte + """
    SELECT
      decade,
      genre_group,
      COUNT(*) AS track_cnt
    FROM base
    WHERE genre_group IS NOT NULL
    GROUP BY decade, genre_group
    ORDER BY decade, genre_group;
    """

    trend_sql = base_cte + """
    SELECT
      decade,
      ROUND(AVG(danceability)::numeric, 3) AS danceability,
      ROUND(AVG(energy)::numeric, 3)       AS energy,
      ROUND(AVG(valence)::numeric, 3)      AS valence,
      ROUND(AVG(acousticness)::numeric, 3) AS acousticness,
      ROUND(AVG(tempo)::numeric, 2)        AS tempo,
      ROUND(AVG(loudness)::numeric, 2)     AS loudness,
      ROUND(AVG(duration_ms)::numeric, 0)  AS duration_ms
    FROM base
    WHERE genre_group IS NOT NULL
      AND (%s = '' OR genre_group = %s)
    GROUP BY decade
    ORDER BY decade;
    """

    summary = {}
    radar_data = {}
    radar_points_str = "300,250 300,250 300,250 300,250 300,250 300,250"
    radar_points_str_all = "300,250 300,250 300,250 300,250 300,250 300,250"
    duration_fmt = "—"
    genres = BIG7[:]
    genre_mix = []
    trend_data = []
    trend_data_all = []

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(genres_sql, (BIG7,))
            db_genres = [r[0] for r in cur.fetchall()]
            # Same order as viewer lower section: BIG7 order
            genres = [g for g in BIG7 if g in db_genres] if db_genres else list(BIG7)

            cur.execute(summary_sql, (BIG7, genre, genre))
            summary_cols = [
                "danceability", "energy", "valence",
                "acousticness", "tempo", "loudness", "duration_ms"
            ]
            summary = fetchone_dict(cur, summary_cols) or {}

            radar_data = {
                "danceability": summary.get("danceability"),
                "energy": summary.get("energy"),
                "valence": summary.get("valence"),
                "acousticness": summary.get("acousticness"),
                "tempo": summary.get("tempo"),
                "loudness": summary.get("loudness"),
                "duration_ms": summary.get("duration_ms"),
            }
            dur_ms = summary.get("duration_ms")
            if dur_ms is not None:
                try:
                    s = int(float(dur_ms))
                    duration_fmt = "%d:%02d" % (s // 60000, (s % 60000) // 1000)
                except (TypeError, ValueError):
                    duration_fmt = "—"
            else:
                duration_fmt = "—"
            # Radar chart 6 axes order: danceability, energy, valence, acousticness, tempo, loudness; values normalized 0–1
            def norm_tempo(v):
                if v is None:
                    return 0
                return max(0, min(1, (float(v) - 50) / 150))

            def norm_loudness(v):
                if v is None:
                    return 0
                return max(0, min(1, (float(v) + 60) / 60))

            radar_values = [
                float(summary.get("danceability") or 0),
                float(summary.get("energy") or 0),
                float(summary.get("valence") or 0),
                float(summary.get("acousticness") or 0),
                norm_tempo(summary.get("tempo")),
                norm_loudness(summary.get("loudness")),
            ]
            import math
            cx, cy, r = 300, 250, 180
            radar_points = []
            for i, v in enumerate(radar_values):
                rad = i * 60 * math.pi / 180
                x = cx + r * v * math.sin(rad)
                y = cy - r * v * math.cos(rad)
                radar_points.append("%.1f,%.1f" % (x, y))
            radar_points_str = " ".join(radar_points)

            # "All" as benchmark: summary with no genre filter, shown in light red
            cur.execute(summary_sql, (BIG7, "", ""))
            summary_all = fetchone_dict(cur, summary_cols) or {}
            radar_values_all = [
                float(summary_all.get("danceability") or 0),
                float(summary_all.get("energy") or 0),
                float(summary_all.get("valence") or 0),
                float(summary_all.get("acousticness") or 0),
                norm_tempo(summary_all.get("tempo")),
                norm_loudness(summary_all.get("loudness")),
            ]
            radar_points_all = []
            for i, v in enumerate(radar_values_all):
                rad = i * 60 * math.pi / 180
                x = cx + r * v * math.sin(rad)
                y = cy - r * v * math.cos(rad)
                radar_points_all.append("%.1f,%.1f" % (x, y))
            radar_points_str_all = " ".join(radar_points_all)

            cur.execute(mix_sql, (BIG7,))
            mix_rows = cur.fetchall()
            genre_mix = []
            trend_data = []

            mix_by_decade = {}
            for decade_int, g, cnt in mix_rows:
                if decade_int is None or g is None:
                    continue
                dlab = decade_label(decade_int)
                mix_by_decade.setdefault(dlab, {k: 0 for k in BIG7})
                if g in mix_by_decade[dlab]:
                    mix_by_decade[dlab][g] += int(cnt)

            for dlab in sorted(mix_by_decade.keys(), key=lambda x: int(x[:-1])):
                counts = mix_by_decade[dlab]
                total = sum(counts.values())
                row = {"decade": dlab}
                if total <= 0:
                    for g in BIG7:
                        row[g] = 0
                else:
                    for g in BIG7:
                        row[g] = round(counts[g] / total, 4)
                genre_mix.append(row)

            cur.execute(trend_sql, (BIG7, trend_genre, trend_genre))
            trend_rows = cur.fetchall()
            trend_data = []
            for r in trend_rows:
                trend_data.append({
                    "decade": decade_label(r[0]),
                    "danceability": float(r[1]) if r[1] is not None else None,
                    "energy": float(r[2]) if r[2] is not None else None,
                    "valence": float(r[3]) if r[3] is not None else None,
                    "acousticness": float(r[4]) if r[4] is not None else None,
                    "tempo": float(r[5]) if r[5] is not None else None,
                    "loudness": float(r[6]) if r[6] is not None else None,
                    "duration_ms": float(r[7]) if r[7] is not None else None,
                })
            cur.execute(trend_sql, (BIG7, "", ""))
            trend_all_rows = cur.fetchall()
            trend_data_all = []
            for r in trend_all_rows:
                trend_data_all.append({
                    "decade": decade_label(r[0]),
                    "danceability": float(r[1]) if r[1] is not None else None,
                    "energy": float(r[2]) if r[2] is not None else None,
                    "valence": float(r[3]) if r[3] is not None else None,
                    "acousticness": float(r[4]) if r[4] is not None else None,
                    "tempo": float(r[5]) if r[5] is not None else None,
                    "loudness": float(r[6]) if r[6] is not None else None,
                    "duration_ms": float(r[7]) if r[7] is not None else None,
                })

    return render_template(
        "analyst.html",
        genres=genres,
        genre=genre,
        trend_genre=trend_genre,
        active_feature=feature,
        feature_tabs=FEATURE_TABS,
        summary=summary,
        radar_data=radar_data,
        radar_points_str=radar_points_str,
        radar_points_str_all=radar_points_str_all,
        duration_fmt=duration_fmt,
        genre_mix=genre_mix,
        trend_data=trend_data,
        trend_data_all=trend_data_all,
    )


@app.route("/analyst/api/trend_data", methods=["GET"])
def analyst_api_trend_data():
    """Return trend_data JSON for a given trend_genre (no page reload)."""
    trend_genre = request.args.get("trend_genre", "").strip()
    if trend_genre and trend_genre not in BIG7:
        trend_genre = ""
    base_cte = f"""
    WITH base AS (
      SELECT DISTINCT
        t.track_id,
        (FLOOR(EXTRACT(YEAR FROM a.release_date)/10)*10)::int AS decade,
        t.duration_ms,
        af.danceability,
        af.energy,
        af.valence,
        af.acousticness,
        af.tempo,
        af.loudness,
        (
          SELECT MIN(ag2.genre_name)
          FROM track_artist ta2
          JOIN artist_genres ag2 ON ag2.artist_id = ta2.artist_id
          WHERE ta2.track_id = t.track_id
            AND ag2.genre_name = ANY(%s)
        ) AS genre_group
      FROM tracks t
      JOIN audio_features af ON af.track_id = t.track_id
      JOIN album_tracks at ON at.track_id = t.track_id
      JOIN albums a ON a.album_id = at.album_id
      WHERE t.status='approved'
        AND a.release_date IS NOT NULL
    )
    """
    trend_sql = base_cte + """
    SELECT
      decade,
      ROUND(AVG(danceability)::numeric, 3) AS danceability,
      ROUND(AVG(energy)::numeric, 3)       AS energy,
      ROUND(AVG(valence)::numeric, 3)      AS valence,
      ROUND(AVG(acousticness)::numeric, 3) AS acousticness,
      ROUND(AVG(tempo)::numeric, 2)        AS tempo,
      ROUND(AVG(loudness)::numeric, 2)     AS loudness,
      ROUND(AVG(duration_ms)::numeric, 0)  AS duration_ms
    FROM base
    WHERE genre_group IS NOT NULL
      AND (%s = '' OR genre_group = %s)
    GROUP BY decade
    ORDER BY decade;
    """
    trend_data = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(trend_sql, (BIG7, trend_genre, trend_genre))
            for r in cur.fetchall():
                trend_data.append({
                    "decade": decade_label(r[0]),
                    "danceability": float(r[1]) if r[1] is not None else None,
                    "energy": float(r[2]) if r[2] is not None else None,
                    "valence": float(r[3]) if r[3] is not None else None,
                    "acousticness": float(r[4]) if r[4] is not None else None,
                    "tempo": float(r[5]) if r[5] is not None else None,
                    "loudness": float(r[6]) if r[6] is not None else None,
                    "duration_ms": float(r[7]) if r[7] is not None else None,
                })
    return jsonify(trend_data)


@app.route("/analyst/api/decade_tracks", methods=["GET"])
def analyst_api_decade_tracks():
    """Return list of tracks for a given decade and genre (track_name, duration_ms, duration_fmt)."""
    trend_genre = request.args.get("trend_genre", "").strip()
    decade_param = request.args.get("decade", "2020").strip()
    if trend_genre and trend_genre not in BIG7:
        trend_genre = ""
    try:
        decade_int = int(decade_param)
    except ValueError:
        decade_int = 2020
    base_cte = f"""
    WITH base AS (
      SELECT DISTINCT
        t.track_id,
        t.track_name,
        t.duration_ms,
        (FLOOR(EXTRACT(YEAR FROM a.release_date)/10)*10)::int AS decade,
        (
          SELECT MIN(ag2.genre_name)
          FROM track_artist ta2
          JOIN artist_genres ag2 ON ag2.artist_id = ta2.artist_id
          WHERE ta2.track_id = t.track_id
            AND ag2.genre_name = ANY(%s)
        ) AS genre_group
      FROM tracks t
      JOIN album_tracks at ON at.track_id = t.track_id
      JOIN albums a ON a.album_id = at.album_id
      WHERE t.status='approved'
        AND a.release_date IS NOT NULL
    )
    """
    sql = base_cte + """
    SELECT track_name, duration_ms
    FROM base
    WHERE genre_group IS NOT NULL AND decade = %s AND (%s = '' OR genre_group = %s)
    ORDER BY duration_ms ASC NULLS LAST, track_name;
    """
    out = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (BIG7, decade_int, trend_genre, trend_genre))
            for name, dur_ms in cur.fetchall():
                if dur_ms is not None:
                    try:
                        s = int(float(dur_ms))
                        fmt = "%d:%02d" % (s // 60000, (s % 60000) // 1000)
                    except (TypeError, ValueError):
                        fmt = "—"
                else:
                    fmt = "—"
                out.append({"track_name": name or "—", "duration_ms": dur_ms, "duration_fmt": fmt})
    return jsonify(out)


# -------------------------
# Admin
# -------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = (request.form.get("username") or "").strip()
        if not email:
            flash("Please enter your email.")
            return render_template("admin_login.html")
        role = "analyst"
        user_id = None
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, email, role FROM users WHERE email = %s",
                    (email,),
                )
                row = cur.fetchone()
                if row:
                    user_id, email, role = row[0], row[1], row[2]
                else:
                    # auto-create demo user for this email
                    cur.execute(
                        "INSERT INTO users (email, role, is_active) VALUES (%s, %s, TRUE) RETURNING user_id",
                        (email, role),
                    )
                    user_id = cur.fetchone()[0]
                    conn.commit()
        session["admin_email"] = email
        session["admin_role"] = role
        if user_id is not None:
            session["admin_user_id"] = user_id
        if (role or "").lower() == "admin":
            return redirect(url_for("admin"))
        return redirect(url_for("admin_gateway"))
    return render_template("admin_login.html")


@app.route("/admin/gateway", methods=["GET"])
def admin_gateway():
    if not session.get("admin_email"):
        return redirect(url_for("admin_login"))
    role = (session.get("admin_role") or "analyst").lower()
    role_display = "Admin" if role == "admin" else "Analyst"
    return render_template(
        "admin_gateway.html",
        email=session.get("admin_email"),
        role_display=role_display,
    )


@app.route("/admin/logout", methods=["GET", "POST"])
def admin_logout():
    session.pop("admin_email", None)
    session.pop("admin_role", None)
    session.pop("admin_user_id", None)
    return redirect(url_for("admin_login"))


@app.route("/admin/manage", methods=["GET"])
def manage_list():
    if not session.get("admin_email"):
        return redirect(url_for("admin_login"))
    # Ensure we always have admin_user_id in session (for Only my submissions),
    # Also handle legacy sessions that logged in before we added auto-create-user logic
    if "admin_user_id" not in session or session.get("admin_user_id") is None:
        email = session.get("admin_email")
        if email:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
                    row = cur.fetchone()
                    if row:
                        session["admin_user_id"] = row[0]

    tab = request.args.get("tab", "songs")
    if tab not in ("songs", "artists", "albums"):
        tab = "songs"
    status = (request.args.get("status") or "all").lower()
    if status not in ("all", "pending", "approved", "rejected"):
        status = "all"
    mine = request.args.get("mine") == "1"
    q = (request.args.get("q") or "").strip()
    decade = (request.args.get("decade") or "all")
    genre = (request.args.get("genre") or "all")
    page_size = 20
    try:
        page = int(request.args.get("page", "1"))
        if page < 1:
            page = 1
    except ValueError:
        page = 1
    offset = (page - 1) * page_size

    items = []
    total = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            if tab == "songs":
                base_where = ["1=1"]
                base_params = []

                if status != "all":
                    base_where.append("t.status = %s")
                    base_params.append(status)

                user_id = session.get("admin_user_id")
                if mine and user_id:
                    base_where.append("t.submitted_by = %s")
                    base_params.append(user_id)

                if decade and decade != "all":
                    try:
                        decade_start = int(decade[:4])
                        decade_end = decade_start + 9
                        base_where.append(
                            "al.release_date IS NOT NULL "
                            "AND EXTRACT(YEAR FROM al.release_date) BETWEEN %s AND %s"
                        )
                        base_params.extend([decade_start, decade_end])
                    except ValueError:
                        pass

                if genre and genre != "all":
                    base_where.append(
                        "EXISTS (SELECT 1 FROM track_genres tg "
                        "WHERE tg.track_id = t.track_id AND tg.genre_name = %s)"
                    )
                    base_params.append(genre)

                # build with search
                where = list(base_where)
                params = list(base_params)
                if q:
                    where.append(
                        "("
                        "LOWER(t.track_name) LIKE %s OR "
                        "LOWER(COALESCE(ar.artist_name,'')) LIKE %s OR "
                        "LOWER(COALESCE(al.album_name,'')) LIKE %s"
                        ")"
                    )
                    like = "%" + q.lower() + "%"
                    params.extend([like, like, like])

                where_sql = " AND ".join(where)
                count_sql = f"""
                    SELECT COUNT(DISTINCT t.track_id)
                    FROM tracks t
                    LEFT JOIN album_tracks at ON at.track_id = t.track_id
                    LEFT JOIN albums al ON al.album_id = at.album_id
                    LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                    LEFT JOIN artists ar ON ar.artist_id = ta.artist_id
                    WHERE {where_sql}
                """
                cur.execute(count_sql, params)
                total = cur.fetchone()[0] or 0

                # If search returns no results, fall back to showing by filters only (status/decade/genre etc.)
                if total == 0 and q:
                    where_sql = " AND ".join(base_where)
                    count_sql = f"""
                        SELECT COUNT(DISTINCT t.track_id)
                        FROM tracks t
                        LEFT JOIN album_tracks at ON at.track_id = t.track_id
                        LEFT JOIN albums al ON al.album_id = at.album_id
                        LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                        LEFT JOIN artists ar ON ar.artist_id = ta.artist_id
                        WHERE {where_sql}
                    """
                    cur.execute(count_sql, base_params)
                    total = cur.fetchone()[0] or 0
                    params = list(base_params)

                list_sql = f"""
                    SELECT
                      t.track_id,
                      t.track_name,
                      COALESCE(ar.artist_name, ''),
                      COALESCE(al.album_name, ''),
                      al.album_image_url,
                      t.status,
                      COALESCE(t.added_at::text, '')
                    FROM tracks t
                    LEFT JOIN album_tracks at ON at.track_id = t.track_id
                    LEFT JOIN albums al ON al.album_id = at.album_id
                    LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                    LEFT JOIN artists ar ON ar.artist_id = ta.artist_id
                    WHERE {where_sql}
                    GROUP BY t.track_id, t.track_name, ar.artist_name, al.album_name, al.album_image_url, t.status, t.added_at
                    ORDER BY
                      CASE t.status
                        WHEN 'pending' THEN 0
                        WHEN 'rejected' THEN 1
                        WHEN 'approved' THEN 2
                        ELSE 3
                      END,
                      CASE WHEN t.added_at IS NULL THEN 1 ELSE 0 END,
                      t.added_at DESC NULLS LAST,
                      t.track_id
                    LIMIT %s OFFSET %s
                """
                cur.execute(list_sql, params + [page_size, offset])
                for r in cur.fetchall():
                    items.append(
                        {
                            "id": r[0],
                            "name": r[1],
                            "artist_name": r[2],
                            "album_name": r[3],
                            "cover_url": r[4],
                            "status": r[5],
                            "last_updated": r[6],
                        }
                    )

            elif tab == "artists":
                base_where = ["1=1"]
                base_params = []

                if status != "all":
                    base_where.append("a.status = %s")
                    base_params.append(status)

                user_id = session.get("admin_user_id")
                if mine and user_id:
                    base_where.append("a.submitted_by = %s")
                    base_params.append(user_id)

                if decade and decade != "all":
                    try:
                        decade_start = int(decade[:4])
                        decade_end = decade_start + 9
                        base_where.append(
                            "EXISTS ("
                            "  SELECT 1 FROM track_artist ta2 "
                            "  JOIN album_tracks at2 ON at2.track_id = ta2.track_id "
                            "  JOIN albums al ON al.album_id = at2.album_id "
                            "  WHERE ta2.artist_id = a.artist_id "
                            "    AND al.release_date IS NOT NULL "
                            "    AND EXTRACT(YEAR FROM al.release_date) BETWEEN %s AND %s"
                            ")"
                        )
                        base_params.extend([decade_start, decade_end])
                    except ValueError:
                        pass

                if genre and genre != "all":
                    base_where.append(
                        "EXISTS (SELECT 1 FROM artist_genres ag "
                        "WHERE ag.artist_id = a.artist_id AND ag.genre_name = %s)"
                    )
                    base_params.append(genre)

                where = list(base_where)
                params = list(base_params)
                if q:
                    where.append("LOWER(a.artist_name) LIKE %s")
                    like = "%" + q.lower() + "%"
                    params.append(like)

                where_sql = " AND ".join(where)
                count_sql = f"SELECT COUNT(*) FROM artists a WHERE {where_sql}"
                cur.execute(count_sql, params)
                total = cur.fetchone()[0] or 0

                if total == 0 and q:
                    where_sql = " AND ".join(base_where)
                    count_sql = f"SELECT COUNT(*) FROM artists a WHERE {where_sql}"
                    cur.execute(count_sql, base_params)
                    total = cur.fetchone()[0] or 0
                    params = list(base_params)

                list_sql = f"""
                    SELECT
                      a.artist_id,
                      a.artist_name,
                      a.status,
                      COALESCE(MIN(t.added_at)::text, '')
                    FROM artists a
                    LEFT JOIN track_artist ta ON ta.artist_id = a.artist_id
                    LEFT JOIN tracks t ON t.track_id = ta.track_id
                    WHERE {where_sql}
                    GROUP BY a.artist_id, a.artist_name, a.status
                    ORDER BY
                      CASE a.status
                        WHEN 'pending' THEN 0
                        WHEN 'rejected' THEN 1
                        WHEN 'approved' THEN 2
                        ELSE 3
                      END,
                      MIN(t.added_at) IS NULL,
                      MIN(t.added_at) DESC,
                      a.artist_name
                    LIMIT %s OFFSET %s
                """
                cur.execute(list_sql, params + [page_size, offset])
                for r in cur.fetchall():
                    items.append(
                        {
                            "id": r[0],
                            "name": r[1],
                            "status": r[2],
                            "last_updated": r[3],
                        }
                    )

            elif tab == "albums":
                base_where = ["1=1"]
                base_params = []

                if status != "all":
                    base_where.append("a.status = %s")
                    base_params.append(status)

                user_id = session.get("admin_user_id")
                if mine and user_id:
                    base_where.append("a.submitted_by = %s")
                    base_params.append(user_id)

                if decade and decade != "all":
                    try:
                        decade_start = int(decade[:4])
                        decade_end = decade_start + 9
                        base_where.append(
                            "a.release_date IS NOT NULL "
                            "AND EXTRACT(YEAR FROM a.release_date) BETWEEN %s AND %s"
                        )
                        base_params.extend([decade_start, decade_end])
                    except ValueError:
                        pass

                if genre and genre != "all":
                    base_where.append(
                        "EXISTS (SELECT 1 FROM album_tracks at2 "
                        "JOIN track_genres tg ON tg.track_id = at2.track_id "
                        "WHERE at2.album_id = a.album_id AND tg.genre_name = %s)"
                    )
                    base_params.append(genre)

                where = list(base_where)
                params = list(base_params)
                if q:
                    where.append(
                        "("
                        "LOWER(a.album_name) LIKE %s OR "
                        "LOWER(COALESCE(ar.artist_name,'')) LIKE %s"
                        ")"
                    )
                    like = "%" + q.lower() + "%"
                    params.extend([like, like])

                where_sql = " AND ".join(where)
                count_sql = f"""
                    SELECT COUNT(DISTINCT a.album_id)
                    FROM albums a
                    LEFT JOIN album_tracks at ON at.album_id = a.album_id
                    LEFT JOIN tracks t ON t.track_id = at.track_id
                    LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                    LEFT JOIN artists ar ON ar.artist_id = ta.artist_id
                    WHERE {where_sql}
                """
                cur.execute(count_sql, params)
                total = cur.fetchone()[0] or 0

                if total == 0 and q:
                    where_sql = " AND ".join(base_where)
                    count_sql = f"""
                        SELECT COUNT(DISTINCT a.album_id)
                        FROM albums a
                        LEFT JOIN album_tracks at ON at.album_id = a.album_id
                        LEFT JOIN tracks t ON t.track_id = at.track_id
                        LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                        LEFT JOIN artists ar ON ar.artist_id = ta.artist_id
                        WHERE {where_sql}
                    """
                    cur.execute(count_sql, base_params)
                    total = cur.fetchone()[0] or 0
                    params = list(base_params)

                list_sql = f"""
                    SELECT
                      a.album_id,
                      a.album_name,
                      a.album_image_url,
                      a.status,
                      COALESCE(MIN(t.added_at)::text, '')
                    FROM albums a
                    LEFT JOIN album_tracks at ON at.album_id = a.album_id
                    LEFT JOIN tracks t ON t.track_id = at.track_id
                    LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                    LEFT JOIN artists ar ON ar.artist_id = ta.artist_id
                    WHERE {where_sql}
                    GROUP BY a.album_id, a.album_name, a.album_image_url, a.status
                    ORDER BY
                      CASE a.status
                        WHEN 'pending' THEN 0
                        WHEN 'rejected' THEN 1
                        WHEN 'approved' THEN 2
                        ELSE 3
                      END,
                      MIN(t.added_at) IS NULL,
                      MIN(t.added_at) DESC,
                      a.album_name
                    LIMIT %s OFFSET %s
                """
                cur.execute(list_sql, params + [page_size, offset])
                for r in cur.fetchall():
                    items.append(
                        {
                            "id": r[0],
                            "name": r[1],
                            "cover_url": r[2],
                            "status": r[3],
                            "last_updated": r[4],
                        }
                    )

    last_page = max(1, (total + page_size - 1) // page_size) if total else 1
    if page > last_page:
        page = last_page
    # simple page window around current page
    start_p = max(1, page - 2)
    end_p = min(last_page, page + 2)
    page_window = list(range(start_p, end_p + 1))

    return render_template(
        "manage_list.html",
        tab=tab,
        status=status,
        mine=mine,
        q=q,
        decade=decade,
        genre=genre,
        items=items,
        total=total,
        page=page,
        last_page=last_page,
        start_index=(offset + 1) if total else 0,
        end_index=min(offset + page_size, total) if total else 0,
        page_window=page_window,
        current_email=session.get("admin_email"),
        current_role=(session.get("admin_role") or "Analyst").capitalize(),
        big7=BIG7,
    )

def _admin_users(cur):
    cur.execute(
        "SELECT user_id, email, role, is_active FROM users ORDER BY email"
    )
    return [
        {"user_id": r[0], "email": r[1], "role": r[2], "is_active": r[3]}
        for r in cur.fetchall()
    ]


def _admin_pending(cur, q=None):
    out = []
    params = ("%" + q + "%",) if q else ()
    cur.execute("""
        SELECT t.track_id, t.track_name, t.submitted_by, u.email
        FROM tracks t
        LEFT JOIN users u ON u.user_id = t.submitted_by
        WHERE t.status = 'pending'
        """ + (" AND t.track_name ILIKE %s" if q else "") + """
        ORDER BY t.track_id
    """, params)
    for r in cur.fetchall():
        out.append({
            "type": "track",
            "id": r[0],
            "title": (r[1] or ""),
            "submitted_by": r[3] or "—",
            "reviewed_by": "—",
            "processed_at": "—",
        })
    cur.execute("""
        SELECT a.album_id, a.album_name, a.submitted_by, u.email
        FROM albums a
        LEFT JOIN users u ON u.user_id = a.submitted_by
        WHERE a.status = 'pending'
        """ + (" AND a.album_name ILIKE %s" if q else "") + """
        ORDER BY a.album_id
    """, params)
    for r in cur.fetchall():
        out.append({
            "type": "album",
            "id": r[0],
            "title": (r[1] or ""),
            "submitted_by": r[3] or "—",
            "reviewed_by": "—",
            "processed_at": "—",
        })
    cur.execute("""
        SELECT a.artist_id, a.artist_name, a.submitted_by, u.email
        FROM artists a
        LEFT JOIN users u ON u.user_id = a.submitted_by
        WHERE a.status = 'pending'
        """ + (" AND a.artist_name ILIKE %s" if q else "") + """
        ORDER BY a.artist_id
    """, params)
    for r in cur.fetchall():
        out.append({
            "type": "artist",
            "id": r[0],
            "title": (r[1] or ""),
            "submitted_by": r[3] or "—",
            "reviewed_by": "—",
            "processed_at": "—",
        })
    return out


def _admin_history(cur, q=None):
    out = []
    search_param = ("%" + q + "%",) if q else ()
    # Prefer query with reviewed_at / reviewed_by; fallback if column missing (e.g. migration not run)
    def fetch_tracks():
        cur.execute(
            """
            SELECT t.track_id, t.track_name, t.status, t.submitted_by, u.email,
                   t.reviewed_by, ru.email AS reviewer_email, t.reviewed_at
            FROM tracks t
            LEFT JOIN users u ON u.user_id = t.submitted_by
            LEFT JOIN users ru ON ru.user_id = t.reviewed_by
            WHERE t.status IN ('approved','rejected')
            """ + (" AND t.track_name ILIKE %s" if q else "") + """
            ORDER BY t.reviewed_at DESC NULLS LAST, t.track_id
            """,
            search_param,
        )
        for r in cur.fetchall():
            out.append({
                "type": "track", "id": r[0], "title": (r[1] or ""),
                "status": r[2] or "pending", "submitted_by": r[4] or "—",
                "reviewed_by": r[6] or "—", "processed_at": _format_reviewed_at(r[7]),
            })

    def fetch_tracks_fallback():
        cur.execute(
            """
            SELECT t.track_id, t.track_name, t.status, t.submitted_by, u.email
            FROM tracks t
            LEFT JOIN users u ON u.user_id = t.submitted_by
            WHERE t.status IN ('approved','rejected')
            """ + (" AND t.track_name ILIKE %s" if q else "") + """
            ORDER BY t.added_at DESC NULLS LAST, t.track_id
            """,
            search_param,
        )
        for r in cur.fetchall():
            out.append({
                "type": "track", "id": r[0], "title": (r[1] or ""),
                "status": r[2] or "pending", "submitted_by": r[4] or "—",
                "reviewed_by": "—", "processed_at": "—",
            })

    def fetch_albums():
        cur.execute(
            """
            SELECT a.album_id, a.album_name, a.status, a.submitted_by, u.email,
                   a.reviewed_by, ru.email AS reviewer_email, a.reviewed_at
            FROM albums a
            LEFT JOIN users u ON u.user_id = a.submitted_by
            LEFT JOIN users ru ON ru.user_id = a.reviewed_by
            WHERE a.status IN ('approved','rejected')
            """ + (" AND a.album_name ILIKE %s" if q else "") + """
            ORDER BY a.reviewed_at DESC NULLS LAST, a.album_id
            """,
            search_param,
        )
        for r in cur.fetchall():
            out.append({
                "type": "album", "id": r[0], "title": (r[1] or ""),
                "status": r[2] or "pending", "submitted_by": r[4] or "—",
                "reviewed_by": r[6] or "—", "processed_at": _format_reviewed_at(r[7]),
            })

    def fetch_albums_fallback():
        cur.execute(
            """
            SELECT a.album_id, a.album_name, a.status, a.submitted_by, u.email
            FROM albums a
            LEFT JOIN users u ON u.user_id = a.submitted_by
            WHERE a.status IN ('approved','rejected')
            """ + (" AND a.album_name ILIKE %s" if q else "") + """
            ORDER BY a.added_at DESC NULLS LAST, a.album_id
            """,
            search_param,
        )
        for r in cur.fetchall():
            out.append({
                "type": "album", "id": r[0], "title": (r[1] or ""),
                "status": r[2] or "pending", "submitted_by": r[4] or "—",
                "reviewed_by": "—", "processed_at": "—",
            })

    def fetch_artists():
        cur.execute(
            """
            SELECT a.artist_id, a.artist_name, a.status, a.submitted_by, u.email,
                   a.reviewed_by, ru.email AS reviewer_email, a.reviewed_at
            FROM artists a
            LEFT JOIN users u ON u.user_id = a.submitted_by
            LEFT JOIN users ru ON ru.user_id = a.reviewed_by
            WHERE a.status IN ('approved','rejected')
            """ + (" AND a.artist_name ILIKE %s" if q else "") + """
            ORDER BY a.reviewed_at DESC NULLS LAST, a.artist_id
            """,
            search_param,
        )
        for r in cur.fetchall():
            out.append({
                "type": "artist", "id": r[0], "title": (r[1] or ""),
                "status": r[2] or "pending", "submitted_by": r[4] or "—",
                "reviewed_by": r[6] or "—", "processed_at": _format_reviewed_at(r[7]),
            })

    def fetch_artists_fallback():
        cur.execute(
            """
            SELECT a.artist_id, a.artist_name, a.status, a.submitted_by, u.email
            FROM artists a
            LEFT JOIN users u ON u.user_id = a.submitted_by
            WHERE a.status IN ('approved','rejected')
            """ + (" AND a.artist_name ILIKE %s" if q else "") + """
            ORDER BY a.added_at DESC NULLS LAST, a.artist_id
            """,
            search_param,
        )
        for r in cur.fetchall():
            out.append({
                "type": "artist", "id": r[0], "title": (r[1] or ""),
                "status": r[2] or "pending", "submitted_by": r[4] or "—",
                "reviewed_by": "—", "processed_at": "—",
            })

    try:
        fetch_tracks()
        fetch_albums()
        fetch_artists()
    except Exception:
        cur.connection.rollback()
        out.clear()
        fetch_tracks_fallback()
        fetch_albums_fallback()
        fetch_artists_fallback()
    return out


def _format_reviewed_at(ts):
    if ts is None:
        return "—"
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d %H:%M")
    return str(ts)[:16] if len(str(ts)) >= 16 else str(ts)


@app.route("/admin", methods=["GET"])
def admin():
    users = []
    pending = []
    history = []
    q_pending = (request.args.get("q_pending") or "").strip()
    q_history = (request.args.get("q_history") or "").strip()
    try:
        page_pending = max(1, int(request.args.get("page_pending", "1")))
    except ValueError:
        page_pending = 1
    try:
        page_history = max(1, int(request.args.get("page_history", "1")))
    except ValueError:
        page_history = 1
    page_size = 20
    show_history = request.args.get("history_open") == "1"
    with get_conn() as conn:
        with conn.cursor() as cur:
            users = _admin_users(cur)
            full_pending = _admin_pending(cur, q=q_pending)
            full_history = _admin_history(cur, q=q_history)
    total_pending = len(full_pending)
    total_history = len(full_history)

    # Pending pagination (20 per page, like manage_list)
    last_page_pending = max(1, (total_pending + page_size - 1) // page_size) if total_pending else 1
    if page_pending > last_page_pending:
        page_pending = last_page_pending
    offset_pending = (page_pending - 1) * page_size
    pending = full_pending[offset_pending : offset_pending + page_size]
    pending_start = (offset_pending + 1) if total_pending else 0
    pending_end = min(offset_pending + page_size, total_pending) if total_pending else 0
    start_p = max(1, page_pending - 2)
    end_p = min(last_page_pending, page_pending + 2)
    pending_page_window = list(range(start_p, end_p + 1))

    # History pagination
    last_page_history = max(1, (total_history + page_size - 1) // page_size) if total_history else 1
    if page_history > last_page_history:
        page_history = last_page_history
    offset_history = (page_history - 1) * page_size
    history = full_history[offset_history : offset_history + page_size]
    history_start = (offset_history + 1) if total_history else 0
    history_end = min(offset_history + page_size, total_history) if total_history else 0
    start_h = max(1, page_history - 2)
    end_h = min(last_page_history, page_history + 2)
    history_page_window = list(range(start_h, end_h + 1))
    return render_template(
        "admin_panel.html",
        users=users,
        pending=pending,
        history=history,
        total_pending=total_pending,
        total_history=total_history,
        page_pending=page_pending,
        page_history=page_history,
        pending_last_page=last_page_pending,
        history_last_page=last_page_history,
        pending_start=pending_start,
        pending_end=pending_end,
        history_start=history_start,
        history_end=history_end,
        pending_page_window=pending_page_window,
        history_page_window=history_page_window,
        page_size=page_size,
        q_pending=q_pending,
        q_history=q_history,
        show_history=show_history,
    )


@app.route("/admin/user/update", methods=["POST"])
def admin_user_update():
    user_id = request.form.get("user_id", "").strip()
    email = (request.form.get("email") or "").strip()
    role = request.form.get("role", "").strip()
    is_active = request.form.get("is_active", "").strip()
    if not user_id or not email or "@" not in email or role not in ("admin", "analyst"):
        flash("Invalid user or role or email.")
        return redirect(url_for("admin"))
    active = is_active.lower() in ("1", "true", "active", "on")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET email = %s, role = %s, is_active = %s WHERE user_id = %s",
                (email, role, active, int(user_id)),
            )
            conn.commit()
    flash("User updated.")
    return redirect(url_for("admin"))


@app.route("/admin/user/create", methods=["POST"])
def admin_user_create():
    email = (request.form.get("email") or "").strip()
    role = request.form.get("role", "").strip()
    is_active = request.form.get("is_active", "").strip()
    if not email or "@" not in email or role not in ("admin", "analyst"):
        flash("Invalid email or role.")
        return redirect(url_for("admin"))
    active = is_active.lower() in ("1", "true", "active", "on")
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (email, password_hash, role, is_active) VALUES (%s, NULL, %s, %s)",
                    (email, role, active),
                )
                conn.commit()
        flash("User created.")
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            flash("A user with this email already exists.")
        else:
            flash("Could not create user.")
    return redirect(url_for("admin"))


@app.route("/admin/review", methods=["POST"])
def admin_review():
    action = request.form.get("action", "").strip()
    item_type = request.form.get("type", "").strip()
    item_id = request.form.get("id", "").strip()
    if action not in ("approve", "reject", "changes", "revert") or item_type not in ("track", "album", "artist"):
        flash("Invalid action or type.")
        return redirect(url_for("admin"))
    table = {"track": "tracks", "album": "albums", "artist": "artists"}[item_type]
    pk = {"track": "track_id", "album": "album_id", "artist": "artist_id"}[item_type]
    reviewer = session.get("admin_user_id")
    with get_conn() as conn:
        with conn.cursor() as cur:
            if action == "revert":
                try:
                    cur.execute(
                        f"UPDATE {table} SET status = 'pending', reviewed_by = NULL, reviewed_at = NULL WHERE {pk} = %s",
                        (item_id,),
                    )
                except Exception:
                    cur.execute(
                        f"UPDATE {table} SET status = 'pending', reviewed_by = NULL WHERE {pk} = %s",
                        (item_id,),
                    )
                conn.commit()
                flash("Reverted to pending.")
            else:
                status = "approved" if action == "approve" else "rejected"
                try:
                    cur.execute(
                        f"UPDATE {table} SET status = %s, reviewed_by = %s, reviewed_at = CURRENT_TIMESTAMP WHERE {pk} = %s",
                        (status, reviewer, item_id),
                    )
                except Exception:
                    cur.execute(
                        f"UPDATE {table} SET status = %s, reviewed_by = %s WHERE {pk} = %s",
                        (status, reviewer, item_id),
                    )
                conn.commit()
                if action == "changes":
                    flash("Changes requested.")
                else:
                    flash(f"Item {action}d.")
    return redirect(url_for("admin"))


@app.route("/artist", methods=["GET"])
def artist():
    artist_id = request.args.get("artist_id", "").strip()
    if not artist_id:
        return render_template(
            "artist.html",
            artist_id=None,
            artist_name=None,
            total_popularity=None,
            rank=None,
            tracks=[],
            albums=[],
            hero_cover=None,
            hero_focus=None,
        )
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Same as viewer Artist: SUM(popularity); same track in multiple albums counts multiple times; full DB, no decade filter
            cur.execute(
                """
                WITH artist_tracks AS (
                  SELECT t.track_id, t.track_name, t.preview_url, t.popularity,
                         al.album_name, al.album_image_url
                  FROM tracks t
                  JOIN track_artist ta ON ta.track_id = t.track_id
                  JOIN album_tracks at ON at.track_id = t.track_id
                  JOIN albums al ON al.album_id = at.album_id
                  WHERE ta.artist_id = %(artist_id)s AND t.status = 'approved'
                ),
                artist_score AS (
                  SELECT SUM(popularity)::int AS total FROM artist_tracks
                ),
                all_scores AS (
                  SELECT ar.artist_id, SUM(t.popularity)::numeric AS score
                  FROM tracks t
                  JOIN track_artist ta ON ta.track_id = t.track_id
                  JOIN artists ar ON ar.artist_id = ta.artist_id
                  JOIN album_tracks at ON at.track_id = t.track_id
                  JOIN albums al ON al.album_id = at.album_id
                  WHERE t.status = 'approved'
                  GROUP BY ar.artist_id
                ),
                ranked AS (
                  SELECT artist_id, ROW_NUMBER() OVER (ORDER BY score DESC NULLS LAST) AS rn
                  FROM all_scores
                )
                SELECT
                  (SELECT artist_name FROM artists WHERE artist_id = %(artist_id)s),
                  (SELECT total FROM artist_score),
                  (SELECT rn FROM ranked WHERE artist_id = %(artist_id)s);
                """,
                {"artist_id": artist_id},
            )
            row = cur.fetchone()
            if not row or row[0] is None:
                return render_template(
                    "artist.html",
                    artist_id=artist_id,
                    artist_name=None,
                    total_popularity=None,
                    rank=None,
                    tracks=[],
                    albums=[],
                    hero_cover=None,
                    hero_focus=None,
                )
            artist_name, total_popularity, rank = row[0], row[1], row[2]

            # Top tracks: all tracks for this artist by popularity desc, one row per track (multiple albums = multiple rows; use DISTINCT ON track_id to keep highest popularity row)
            cur.execute(
                """
                SELECT DISTINCT ON (t.track_id)
                  t.track_id,
                  t.track_name,
                  al.album_id,
                  al.album_name,
                  al.album_image_url,
                  t.popularity,
                  t.preview_url
                FROM tracks t
                JOIN track_artist ta ON ta.track_id = t.track_id
                JOIN album_tracks at ON at.track_id = t.track_id
                JOIN albums al ON al.album_id = at.album_id
                WHERE ta.artist_id = %(artist_id)s AND t.status = 'approved'
                ORDER BY t.track_id, t.popularity DESC NULLS LAST
                """,
                {"artist_id": artist_id},
            )
            track_rows = cur.fetchall()
            # Order by popularity desc, take top 10
            tracks = sorted(
                [
                    {
                        "track_id": r[0],
                        "track_name": r[1],
                        "album_id": r[2],
                        "album_name": r[3],
                        "album_image_url": r[4],
                        "popularity": r[5],
                        "preview_url": r[6],
                    }
                    for r in track_rows
                ],
                key=lambda x: (x["popularity"] or 0),
                reverse=True,
            )[:10]

            # Albums: albums this artist appears on
            cur.execute(
                """
                SELECT DISTINCT a.album_id, a.album_name, a.release_date, a.album_image_url
                FROM albums a
                JOIN album_tracks at ON at.album_id = a.album_id
                JOIN track_artist ta ON ta.track_id = at.track_id
                WHERE ta.artist_id = %(artist_id)s AND a.status = 'approved'
                ORDER BY a.release_date DESC NULLS LAST, a.album_name
                """,
                {"artist_id": artist_id},
            )
            album_rows = cur.fetchall()
            albums_raw = [
                {
                    "album_id": r[0],
                    "album_name": r[1],
                    "release_date": str(r[2])[:4] if r[2] else None,
                    "album_image_url": r[3],
                }
                for r in album_rows
            ]
            # Dedupe by album name + year; same album (e.g. deluxe/same year) shown once (keep first row; already ordered by release_date DESC)
            seen_key = set()
            albums = []
            for a in albums_raw:
                key = (a["album_name"] or "", a["release_date"] or "")
                if key not in seen_key:
                    seen_key.add(key)
                    albums.append(a)

            # Artist page hero: use latest album cover (albums already ordered by release_date DESC)
            hero_cover = (albums[0]["album_image_url"] if albums else None) or (tracks[0]["album_image_url"] if tracks else None)
            # Auto face detection so face lies in hero crop; fallback to centered if none or failure
            hero_focus = get_face_focus(hero_cover) if hero_cover else None

    return render_template(
        "artist.html",
        artist_id=artist_id,
        artist_name=artist_name,
        total_popularity=total_popularity,
        rank=rank,
        tracks=tracks,
        albums=albums,
        hero_cover=hero_cover,
        hero_focus=hero_focus,
    )


@app.route("/analyst/edit/api/search_artists", methods=["GET"])
def analyst_edit_search_artists():
    q = (request.args.get("q") or "").strip()[:80]
    if not q:
        return jsonify([])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT artist_id, artist_name FROM artists
                   WHERE status = 'approved' AND lower(artist_name) LIKE %s
                   ORDER BY artist_name LIMIT 15""",
                ("%" + q.lower() + "%",),
            )
            rows = cur.fetchall()
    return jsonify([{"artist_id": r[0], "artist_name": r[1]} for r in rows])


@app.route("/analyst/edit/api/search_albums", methods=["GET"])
def analyst_edit_search_albums():
    q = (request.args.get("q") or "").strip()[:80]
    if not q:
        return jsonify([])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT album_id, album_name FROM albums
                   WHERE status = 'approved' AND lower(album_name) LIKE %s
                   ORDER BY album_name LIMIT 15""",
                ("%" + q.lower() + "%",),
            )
            rows = cur.fetchall()
    return jsonify([{"album_id": r[0], "album_name": r[1]} for r in rows])


@app.route("/analyst/edit/artist", methods=["GET", "POST"])
def analyst_edit_artist():
    is_admin = (session.get("admin_role") or "").lower() == "admin"
    review_mode = request.method == "GET" and request.args.get("review") == "1"
    return_history_open = request.args.get("history_open") == "1"
    artist_id = request.args.get("artist_id", "").strip() if request.method == "GET" else request.form.get("artist_id", "").strip()

    if request.method == "POST":
        if request.form.get("delete") == "1":
            if artist_id:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM artists WHERE artist_id = %s", (artist_id,))
                        conn.commit()
                flash("Artist deleted.")
            return redirect(url_for("manage_list", tab="artists"))
        artist_name = (request.form.get("artist_name") or "").strip()
        if not artist_name:
            flash("Artist name is required.")
            return redirect(request.url)
        genres = [g for g in request.form.getlist("genre") if g in BIG7]
        with get_conn() as conn:
            with conn.cursor() as cur:
                submitted_by = session.get("admin_user_id")
                if artist_id:
                    cur.execute(
                        "UPDATE artists SET artist_name = %s, submitted_by = COALESCE(%s, submitted_by) WHERE artist_id = %s",
                        (artist_name, submitted_by, artist_id),
                    )
                else:
                    artist_id = f"local_{uuid.uuid4().hex}"
                    cur.execute(
                        "INSERT INTO artists (artist_id, artist_name, status, added_at, submitted_by) VALUES (%s,%s,'pending',CURRENT_DATE,%s)",
                        (artist_id, artist_name, submitted_by),
                    )
                cur.execute("DELETE FROM artist_genres WHERE artist_id = %s", (artist_id,))
                for g in genres:
                    cur.execute(
                        "INSERT INTO artist_genres (artist_id, genre_name) VALUES (%s,%s) ON CONFLICT (artist_id, genre_name) DO NOTHING",
                        (artist_id, g),
                    )
                if genres:
                    cur.execute(
                        "INSERT INTO artist_primary_genre (artist_id, genre_name) VALUES (%s,%s) "
                        "ON CONFLICT (artist_id) DO UPDATE SET genre_name = EXCLUDED.genre_name",
                        (artist_id, genres[0]),
                    )
                else:
                    cur.execute("DELETE FROM artist_primary_genre WHERE artist_id = %s", (artist_id,))
                conn.commit()
        flash("Artist saved.")
        return redirect(url_for("analyst_edit_artist", artist_id=artist_id))

    artist = None
    artist_genres = []
    artist_albums = []
    if artist_id:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT artist_id, artist_name, status FROM artists WHERE artist_id = %s",
                    (artist_id,),
                )
                row = cur.fetchone()
                if row:
                    artist = {
                        "artist_id": row[0],
                        "artist_name": row[1],
                        "status": row[2],
                    }
                    cur.execute("SELECT genre_name FROM artist_genres WHERE artist_id = %s", (artist_id,))
                    artist_genres = [r[0] for r in cur.fetchall()]
                    cur.execute(
                        """
                        SELECT a.album_id, a.album_name, a.release_date
                        FROM albums a
                        JOIN album_tracks at ON at.album_id = a.album_id
                        JOIN tracks t ON t.track_id = at.track_id
                        JOIN track_artist ta ON ta.track_id = t.track_id
                        WHERE ta.artist_id = %s
                        GROUP BY a.album_id, a.album_name, a.release_date
                        ORDER BY a.release_date DESC NULLS LAST, a.album_name
                        """,
                        (artist_id,),
                    )
                    artist_albums = [
                        {
                            "album_id": r[0],
                            "album_name": r[1],
                            "release_year": str(r[2])[:4] if r[2] else None,
                        }
                        for r in cur.fetchall()
                    ]

    return render_template(
        "analyst_edit.html",
        mode="artist",
        artist=artist,
        artist_genres=artist_genres,
        artist_albums=artist_albums,
        big7=BIG7,
        review_mode=review_mode,
        is_admin=is_admin,
        return_history_open=return_history_open,
    )


@app.route("/analyst/edit/album", methods=["GET", "POST"])
def analyst_edit_album():
    is_admin = (session.get("admin_role") or "").lower() == "admin"
    review_mode = request.method == "GET" and request.args.get("review") == "1"
    return_history_open = request.args.get("history_open") == "1"
    album_id = request.args.get("album_id", "").strip() if request.method == "GET" else request.form.get("album_id", "").strip()

    if request.method == "POST":
        if request.form.get("delete") == "1":
            if album_id:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM albums WHERE album_id = %s", (album_id,))
                        conn.commit()
                flash("Album deleted.")
            return redirect(url_for("manage_list", tab="albums"))
        album_name = (request.form.get("album_name") or "").strip()
        if not album_name:
            flash("Album name is required.")
            return redirect(request.url)
        release_date_raw = (request.form.get("release_date") or "").strip()
        album_image_url = (request.form.get("album_image_url") or "").strip() or None
        main_artist_id = (request.form.get("main_artist_id") or "").strip()
        with get_conn() as conn:
            with conn.cursor() as cur:
                submitted_by = session.get("admin_user_id")
                if release_date_raw:
                    try:
                        cur.execute("SELECT %s::date", (release_date_raw,))
                        release_date = release_date_raw
                    except Exception:
                        release_date = None
                else:
                    release_date = None
                if album_id:
                    cur.execute(
                        "UPDATE albums SET album_name=%s, release_date=COALESCE(%s, release_date), "
                        "album_image_url=COALESCE(%s, album_image_url), submitted_by=COALESCE(%s, submitted_by) "
                        "WHERE album_id=%s",
                        (album_name, release_date, album_image_url, submitted_by, album_id),
                    )
                else:
                    album_id = f"local_{uuid.uuid4().hex}"
                    cur.execute(
                        "INSERT INTO albums (album_id, album_name, release_date, album_image_url, status, added_at, submitted_by) "
                        "VALUES (%s,%s,%s,%s,'pending',CURRENT_DATE,%s)",
                        (album_id, album_name, release_date, album_image_url, submitted_by),
                    )
                # If analyst chose a new main artist, relink all tracks under this album to that artist
                if main_artist_id:
                    cur.execute(
                        """
                        UPDATE track_artist
                        SET artist_id = %s
                        WHERE track_id IN (
                          SELECT track_id FROM album_tracks WHERE album_id = %s
                        )
                        """,
                        (main_artist_id, album_id),
                    )
                conn.commit()
        flash("Album saved.")
        return redirect(url_for("analyst_edit_album", album_id=album_id))

    album = None
    album_tracks = []
    main_artist = None
    if album_id:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT album_id, album_name, release_date, album_image_url, status FROM albums WHERE album_id = %s",
                    (album_id,),
                )
                row = cur.fetchone()
                if row:
                    album = {
                        "album_id": row[0],
                        "album_name": row[1],
                        "release_date": str(row[2]) if row[2] else "",
                        "album_image_url": row[3],
                        "status": row[4],
                    }
                    cur.execute(
                        """
                        WITH at_tracks AS (
                          SELECT t.track_id, t.popularity, ta.artist_id
                          FROM album_tracks at
                          JOIN tracks t ON t.track_id = at.track_id
                          LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                          WHERE at.album_id = %s
                        ),
                        artist_score AS (
                          SELECT artist_id, SUM(popularity)::numeric AS artist_pop
                          FROM at_tracks WHERE artist_id IS NOT NULL
                          GROUP BY artist_id
                        ),
                        best_artist AS (
                          SELECT ar.artist_id, ar.artist_name
                          FROM artist_score s
                          JOIN artists ar ON ar.artist_id = s.artist_id
                          ORDER BY s.artist_pop DESC NULLS LAST
                          LIMIT 1
                        )
                        SELECT artist_id, artist_name FROM best_artist
                        """,
                        (album_id,),
                    )
                    ba = cur.fetchone()
                    if ba:
                        main_artist = {"artist_id": ba[0], "artist_name": ba[1]}
                    cur.execute(
                        """
                        SELECT
                          at.disc_number,
                          at.track_number,
                          t.track_name,
                          COALESCE(ar.artist_name, '') AS artist_name
                        FROM album_tracks at
                        JOIN tracks t ON t.track_id = at.track_id
                        LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                        LEFT JOIN artists ar ON ar.artist_id = ta.artist_id
                        WHERE at.album_id = %s
                        ORDER BY COALESCE(at.disc_number, 1), COALESCE(at.track_number, 1), t.track_name
                        """,
                        (album_id,),
                    )
                    album_tracks = [
                        {
                            "disc_number": r[0],
                            "track_number": r[1],
                            "track_name": r[2],
                            "artist_name": r[3],
                        }
                        for r in cur.fetchall()
                    ]

    return render_template(
        "analyst_edit.html",
        mode="album",
        album=album,
        album_tracks=album_tracks,
        main_artist=main_artist,
        big7=BIG7,
        review_mode=review_mode,
        is_admin=is_admin,
        return_history_open=return_history_open,
    )

@app.route("/analyst/edit", methods=["GET", "POST"])
def analyst_edit():
    is_admin = (session.get("admin_role") or "").lower() == "admin"
    review_mode = request.method == "GET" and request.args.get("review") == "1"
    return_history_open = request.args.get("history_open") == "1"
    track_id = request.args.get("track_id", "").strip() if request.method == "GET" else request.form.get("track_id", "").strip()

    # POST: save or delete
    if request.method == "POST":
        if request.form.get("delete") == "1":
            if track_id:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM tracks WHERE track_id = %s", (track_id,))
                        conn.commit()
                flash("Track deleted.")
            return redirect(url_for("manage_list", tab="songs"))
        # Save
        if not track_id:
            flash("No track selected to save.")
            return redirect(url_for("manage_list", tab="songs"))
        track_name = request.form.get("track_name", "").strip() or None
        duration_ms = request.form.get("duration_ms", "").strip()
        duration_ms = int(duration_ms) if duration_ms.isdigit() else None
        preview_url = request.form.get("preview_url", "").strip() or None
        release_date = request.form.get("release_date", "").strip() or None
        album_cover_url = request.form.get("album_cover_url", "").strip() or None
        explicit_raw = request.form.get("explicit")
        # checkbox: value \"1\" when checked, missing when not
        explicit_flag = True if explicit_raw == "1" else False
        genres = [g for g in request.form.getlist("genre") if g in BIG7]

        # Audio features manual override (optional)
        def _parse_float(val):
            val = (val or "").strip()
            if not val:
                return None
            try:
                return float(val)
            except ValueError:
                return None

        danceability = _parse_float(request.form.get("danceability"))
        energy = _parse_float(request.form.get("energy"))
        valence = _parse_float(request.form.get("valence"))
        acousticness = _parse_float(request.form.get("acousticness"))
        loudness = _parse_float(request.form.get("loudness"))
        tempo = _parse_float(request.form.get("tempo"))

        disc_number_raw = (request.form.get("disc_number") or "").strip()
        disc_number = int(disc_number_raw) if disc_number_raw.isdigit() else None
        track_number_raw = (request.form.get("track_number") or "").strip()
        track_number = int(track_number_raw) if track_number_raw.isdigit() else None

        with get_conn() as conn:
            with conn.cursor() as cur:
                submitted_by = session.get("admin_user_id")
                cur.execute(
                    "UPDATE tracks SET track_name=COALESCE(%s,track_name), duration_ms=COALESCE(%s,duration_ms), preview_url=COALESCE(%s,preview_url), explicit=%s, submitted_by=COALESCE(%s, submitted_by) WHERE track_id=%s",
                    (track_name, duration_ms, preview_url, explicit_flag, submitted_by, track_id),
                )
                if release_date or album_cover_url is not None:
                    cur.execute(
                        """UPDATE albums SET release_date=COALESCE(%s,release_date), album_image_url=COALESCE(%s,album_image_url)
                           WHERE album_id = (SELECT album_id FROM album_tracks WHERE track_id=%s LIMIT 1)""",
                        (release_date if release_date else None, album_cover_url if album_cover_url else None, track_id),
                    )
                # Upsert audio features; keep other columns nullable
                cur.execute(
                    """
                    INSERT INTO audio_features (track_id, danceability, energy, valence, acousticness, loudness, tempo)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (track_id) DO UPDATE SET
                      danceability = EXCLUDED.danceability,
                      energy       = EXCLUDED.energy,
                      valence      = EXCLUDED.valence,
                      acousticness = EXCLUDED.acousticness,
                      loudness     = EXCLUDED.loudness,
                      tempo        = EXCLUDED.tempo
                    """,
                    (track_id, danceability, energy, valence, acousticness, loudness, tempo),
                )

                # Update relations: track_artist
                cur.execute("DELETE FROM track_artist WHERE track_id = %s", (track_id,))
                artist_id = (request.form.get("artist_id") or "").strip()
                if artist_id:
                    cur.execute(
                        "INSERT INTO track_artist (track_id, artist_id) VALUES (%s,%s) ON CONFLICT (track_id, artist_id) DO NOTHING",
                        (track_id, artist_id),
                    )

                # Update relations: album_tracks (album + disc / track #)
                cur.execute("DELETE FROM album_tracks WHERE track_id = %s", (track_id,))
                album_id = (request.form.get("album_id") or "").strip()
                if album_id:
                    cur.execute(
                        """
                        INSERT INTO album_tracks (album_id, track_id, disc_number, track_number)
                        VALUES (%s,%s,%s,%s)
                        ON CONFLICT (album_id, track_id) DO UPDATE SET
                          disc_number = EXCLUDED.disc_number,
                          track_number = EXCLUDED.track_number
                        """,
                        (album_id, track_id, disc_number, track_number),
                    )

                cur.execute("DELETE FROM track_genres WHERE track_id = %s", (track_id,))
                for g in genres:
                    cur.execute(
                        "INSERT INTO track_genres (track_id, genre_name) VALUES (%s,%s) ON CONFLICT (track_id, genre_name) DO NOTHING",
                        (track_id, g),
                    )
                conn.commit()
        flash("Track saved.")
        return redirect(url_for("analyst"))

    # GET: load track for edit or show empty (add mode)
    track = None
    if track_id:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      t.track_id,
                      t.track_name,
                      t.duration_ms,
                      t.preview_url,
                      t.explicit,
                      t.status,
                      ar.artist_id,
                      ar.artist_name,
                      al.album_id,
                      al.album_name,
                      al.release_date,
                      al.album_image_url,
                      at.disc_number,
                      at.track_number,
                      af.danceability,
                      af.energy,
                      af.valence,
                      af.acousticness,
                      af.loudness,
                      af.tempo
                    FROM tracks t
                    LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                    LEFT JOIN artists ar ON ar.artist_id = ta.artist_id
                    LEFT JOIN album_tracks at ON at.track_id = t.track_id
                    LEFT JOIN albums al ON al.album_id = at.album_id
                    LEFT JOIN audio_features af ON af.track_id = t.track_id
                    WHERE t.track_id = %s
                    LIMIT 1
                    """,
                    (track_id,),
                )
                row = cur.fetchone()
                if row:
                    cur.execute("SELECT genre_name FROM track_genres WHERE track_id = %s", (track_id,))
                    genre_names = [r[0] for r in cur.fetchall()]
                    track = {
                        "track_id": row[0],
                        "track_name": row[1],
                        "duration_ms": row[2],
                        "preview_url": row[3],
                        "explicit": row[4],
                        "status": row[5],
                        "artist_id": row[6],
                        "artist_name": row[7],
                        "album_id": row[8],
                        "album_name": row[9],
                        "release_date": str(row[10]) if row[10] else None,
                        "album_image_url": row[11],
                        "disc_number": row[12],
                        "track_number": row[13],
                        "danceability": row[14],
                        "energy": row[15],
                        "valence": row[16],
                        "acousticness": row[17],
                        "loudness": row[18],
                        "tempo": row[19],
                        "genre_names": genre_names,
                    }

    return render_template("analyst_edit.html", track=track, big7=BIG7, review_mode=review_mode, is_admin=is_admin, return_history_open=return_history_open)


if __name__ == "__main__":
    app.run(debug=True, port=5001)