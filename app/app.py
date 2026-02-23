# app/app.py
"""
Viewer 各表排序/排名逻辑（/viewer 的 Song / Artist / Album Top Chart）：

- Song Top Chart
  - 筛选：t.status = 'approved'，且按专辑 release_date 做 decade 筛选（可选）。
  - 去重：DISTINCT ON (t.track_id)，每首歌只保留一行（多专辑/多艺人时取 popularity 最高的一行）。
  - 排序：按 popularity DESC NULLS LAST，取前 50 条（前端只展示前 10）。

- Artist Top Chart
  - 筛选：与 Song 一致，按专辑 release_date 做 decade 筛选。
  - 热度：同一首歌在不同专辑出现多次则算多次（不去重），按艺人 SUM(popularity) 得到 score。
  - 每人 TOP TRACK：该艺人在筛选集合内 popularity 最高的那一首（ROW_NUMBER() rn=1）。
  - 排序：按 score（SUM(popularity)）DESC NULLS LAST，取前 10。

- Album Top Chart
  - 筛选：与 Song 一致，t.status = 'approved' 且按专辑 release_date 做 decade 筛选。
  - 热度：专辑热度 = 专辑内所有歌曲的热度之和（每首歌只计一次，SUM(t.popularity) 按专辑聚合）。
  - 专辑艺人：取该专辑内总热度最高的艺人（避免 MIN(artist_name) 按字母序误显示 feat. 艺人）。
  - 排序：按专辑热度 DESC NULLS LAST，取前 10。

- EXPLORE BY GENRES（/viewer 下半部分）
  - 歌曲/专辑/艺人 均按「最主流」genre 筛选，与 ETL 一致：歌曲/专辑用 track_genres（CSV 该 track 的 genre 列表第一个映射到 Big-7），艺人用 artist_primary_genre（该艺人热度最高且存在 track_genres 的曲目的 primary genre）。
  - API：GET /viewer/api/genre_chart?genre=Pop&decade=all&type=songs|artists|albums，逻辑与上述 Song/Artist/Album 一致，genre 筛选：songs/albums 用 track_genres，artists 用 artist_primary_genre；只返回 top 5。
  - 前端保持现有排版，按所选 genre、decade、Songs/Artists/Albums 请求 API 并渲染 grid。

- 歌手页 /artist?artist_id=xxx
  - 与 Viewer Artist 一致：总热度 = SUM(popularity)（该艺人所有曲目，同一首歌在不同专辑出现多次则算多次）；Rank = 全库艺人按该 SUM 降序排名。
  - 右上角：第一行显示 # {rank}，第二行显示 Popularity {total_popularity}。TOP TRACKS、ALBUMS 由后端查询渲染。
"""
import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import psycopg2
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
    return psycopg2.connect(
        host=os.getenv("DB_HOST") or "localhost",
        port=int(os.getenv("DB_PORT") or 5432),
        dbname=os.getenv("DB_NAME") or "musicbox",
        user=os.getenv("DB_USER") or os.getenv("USER"),
        password=os.getenv("DB_PASSWORD") or None,
    )

app = Flask(__name__)
app.secret_key = "dev-secret"

# Big-7 categories (same as your ETL grouping output)
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


# 歌手页 hero 背景：按专辑图人脸位置计算 background-position（缓存按 URL）
_hero_focus_cache = {}


def get_face_focus(image_url, timeout=4):
    """根据专辑图检测人脸，返回 CSS background-position 百分比，使人脸落在视区内。无脸或失败返回 None。"""
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
        # 使人脸中心对应到视区相对位置，便于统一裁切
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

    if chart == "song":
        song_sql = """
        SELECT *
        FROM (
          SELECT DISTINCT ON (t.track_id)
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
            AND (%(decade_start)s IS NULL OR (
                  al.release_date IS NOT NULL
                  AND EXTRACT(YEAR FROM al.release_date)
                      BETWEEN %(decade_start)s AND %(decade_end)s
            ))
          ORDER BY t.track_id, t.popularity DESC NULLS LAST
        ) AS x
        ORDER BY x.popularity DESC NULLS LAST
        LIMIT 50;
        """
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    song_sql,
                    {"decade_start": decade_start, "decade_end": decade_end},
                )
                rows = cur.fetchall()

    elif chart == "artist":
        # 与 Song Top Chart 一致：按专辑 release_date 做 decade 筛选。同一首歌在不同专辑出现多次则算多次热度（不去重）
        artist_sql = """
        WITH decade_tracks AS (
          SELECT t.track_id, t.track_name, t.preview_url, t.popularity, ar.artist_id, ar.artist_name
          FROM tracks t
          JOIN track_artist ta ON ta.track_id = t.track_id
          JOIN artists ar ON ar.artist_id = ta.artist_id
          JOIN album_tracks at ON at.track_id = t.track_id
          JOIN albums al ON al.album_id = at.album_id
          WHERE t.status = 'approved'
            AND (%(decade_start)s IS NULL OR (
                  al.release_date IS NOT NULL
                  AND EXTRACT(YEAR FROM al.release_date)
                      BETWEEN %(decade_start)s AND %(decade_end)s
            ))
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
            with conn.cursor() as cur:
                cur.execute(
                    artist_sql,
                    {"decade_start": decade_start, "decade_end": decade_end},
                )
                rows = cur.fetchall()

    elif chart == "album":
        # 专辑热度 = 专辑内所有歌曲热度之和（每首歌只计一次）；专辑艺人取该专辑内总热度最高的艺人
        album_sql = """
        WITH album_total AS (
          SELECT a.album_id, a.album_name, a.album_image_url,
                 SUM(t.popularity) AS album_total
          FROM albums a
          JOIN album_tracks at ON at.album_id = a.album_id
          JOIN tracks t ON t.track_id = at.track_id
          WHERE t.status = 'approved'
            AND (%(decade_start)s IS NULL OR (
                  a.release_date IS NOT NULL
                  AND EXTRACT(YEAR FROM a.release_date)
                      BETWEEN %(decade_start)s AND %(decade_end)s
            ))
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
            AND (%(decade_start)s IS NULL OR (
                  a.release_date IS NOT NULL
                  AND EXTRACT(YEAR FROM a.release_date)
                      BETWEEN %(decade_start)s AND %(decade_end)s
            ))
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
            with conn.cursor() as cur:
                cur.execute(
                    album_sql,
                    {"decade_start": decade_start, "decade_end": decade_end},
                )
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
    params = {
        "genre": genre,
        "decade_start": decade_start,
        "decade_end": decade_end,
    }
    items = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            if chart_type == "songs":
                sql = """
                SELECT *
                FROM (
                  SELECT DISTINCT ON (t.track_id)
                    t.track_id, t.track_name, t.popularity, t.preview_url,
                    ar.artist_id, ar.artist_name, al.album_id, al.album_name, al.album_image_url
                  FROM tracks t
                  JOIN track_genres tg ON tg.track_id = t.track_id AND tg.genre_name = %(genre)s
                  JOIN track_artist ta ON ta.track_id = t.track_id
                  JOIN artists ar ON ar.artist_id = ta.artist_id
                  JOIN album_tracks at ON at.track_id = t.track_id
                  JOIN albums al ON al.album_id = at.album_id
                  WHERE t.status = 'approved'
                    AND (%(decade_start)s IS NULL OR (
                          al.release_date IS NOT NULL
                          AND EXTRACT(YEAR FROM al.release_date)
                              BETWEEN %(decade_start)s AND %(decade_end)s
                    ))
                  ORDER BY t.track_id, t.popularity DESC NULLS LAST
                ) AS x
                ORDER BY x.popularity DESC NULLS LAST
                LIMIT 5;
                """
                cur.execute(sql, params)
                rows = cur.fetchall()
                items = [
                    {
                        "rank": i + 1,
                        "track_id": r[0],
                        "track_name": r[1],
                        "artist_id": r[4],
                        "artist_name": r[5],
                        "album_id": r[6],
                        "album_name": r[7],
                        "album_image_url": r[8],
                        "preview_url": r[3],
                    }
                    for i, r in enumerate(rows)
                ]
            elif chart_type == "artists":
                sql = """
                WITH decade_tracks AS (
                  SELECT t.track_id, t.track_name, t.preview_url, t.popularity, ar.artist_id, ar.artist_name, al.album_image_url
                  FROM tracks t
                  JOIN track_artist ta ON ta.track_id = t.track_id
                  JOIN artists ar ON ar.artist_id = ta.artist_id
                  JOIN artist_primary_genre apg ON apg.artist_id = ar.artist_id AND apg.genre_name = %(genre)s
                  JOIN album_tracks at ON at.track_id = t.track_id
                  JOIN albums al ON al.album_id = at.album_id
                  WHERE t.status = 'approved'
                    AND (%(decade_start)s IS NULL OR (
                          al.release_date IS NOT NULL
                          AND EXTRACT(YEAR FROM al.release_date)
                              BETWEEN %(decade_start)s AND %(decade_end)s
                    ))
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
                sql = """
                WITH album_total AS (
                  SELECT a.album_id, a.album_name, a.album_image_url,
                         SUM(t.popularity) AS album_total
                  FROM albums a
                  JOIN album_tracks at ON at.album_id = a.album_id
                  JOIN tracks t ON t.track_id = at.track_id
                  WHERE t.status = 'approved'
                    AND (%(decade_start)s IS NULL OR (
                          a.release_date IS NOT NULL
                          AND EXTRACT(YEAR FROM a.release_date)
                              BETWEEN %(decade_start)s AND %(decade_end)s
                    ))
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
                    AND (%(decade_start)s IS NULL OR (
                          a.release_date IS NOT NULL
                          AND EXTRACT(YEAR FROM a.release_date)
                              BETWEEN %(decade_start)s AND %(decade_end)s
                    ))
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
            # 基本信息 + 总热度 + 主艺人（专辑内总 popularity 最高的艺人）
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

            # 专辑内所有曲目（按 track_number 排序，无编号则靠后按曲名）
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
            # 与 viewer 下方顺序一致：按 BIG7 顺序
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
            # 雷达图 6 轴顺序：danceability, energy, valence, acousticness, tempo, loudness；值归一化 0–1
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

            # All 作为 benchmark：无 genre 筛选的 summary，浅红显示
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
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT email, role FROM users WHERE email = %s",
                    (email,),
                )
                row = cur.fetchone()
                if row:
                    email, role = row[0], row[1]
        session["admin_email"] = email
        session["admin_role"] = role
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
    return redirect(url_for("admin_login"))


def _admin_users(cur):
    cur.execute(
        "SELECT user_id, email, role, is_active FROM users ORDER BY email"
    )
    return [
        {"user_id": r[0], "email": r[1], "role": r[2], "is_active": r[3]}
        for r in cur.fetchall()
    ]


def _admin_pending(cur):
    out = []
    cur.execute("""
        SELECT t.track_id, t.track_name, t.submitted_by, u.email
        FROM tracks t
        LEFT JOIN users u ON u.user_id = t.submitted_by
        WHERE t.status = 'pending'
        ORDER BY t.track_id
    """)
    for r in cur.fetchall():
        out.append({
            "type": "track",
            "id": r[0],
            "title": "Track: " + (r[1] or ""),
            "submitted_by": r[3] or "—",
        })
    cur.execute("""
        SELECT a.album_id, a.album_name, a.submitted_by, u.email
        FROM albums a
        LEFT JOIN users u ON u.user_id = a.submitted_by
        WHERE a.status = 'pending'
        ORDER BY a.album_id
    """)
    for r in cur.fetchall():
        out.append({
            "type": "album",
            "id": r[0],
            "title": "Album: " + (r[1] or ""),
            "submitted_by": r[3] or "—",
        })
    cur.execute("""
        SELECT a.artist_id, a.artist_name, a.submitted_by, u.email
        FROM artists a
        LEFT JOIN users u ON u.user_id = a.submitted_by
        WHERE a.status = 'pending'
        ORDER BY a.artist_id
    """)
    for r in cur.fetchall():
        out.append({
            "type": "artist",
            "id": r[0],
            "title": "Artist: " + (r[1] or ""),
            "submitted_by": r[3] or "—",
        })
    return out


@app.route("/admin", methods=["GET"])
def admin():
    users = []
    pending = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            users = _admin_users(cur)
            pending = _admin_pending(cur)
    return render_template(
        "admin_panel.html",
        users=users,
        pending=pending,
    )


@app.route("/admin/user/update", methods=["POST"])
def admin_user_update():
    user_id = request.form.get("user_id", "").strip()
    role = request.form.get("role", "").strip()
    is_active = request.form.get("is_active", "").strip()
    if not user_id or role not in ("admin", "analyst"):
        flash("Invalid user or role.")
        return redirect(url_for("admin"))
    active = is_active.lower() in ("1", "true", "active", "on")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET role = %s, is_active = %s WHERE user_id = %s",
                (role, active, int(user_id)),
            )
            conn.commit()
    flash("User updated.")
    return redirect(url_for("admin"))


@app.route("/admin/review", methods=["POST"])
def admin_review():
    action = request.form.get("action", "").strip()
    item_type = request.form.get("type", "").strip()
    item_id = request.form.get("id", "").strip()
    if action not in ("approve", "reject") or item_type not in ("track", "album", "artist"):
        flash("Invalid action or type.")
        return redirect(url_for("admin"))
    table = {"track": "tracks", "album": "albums", "artist": "artists"}[item_type]
    pk = {"track": "track_id", "album": "album_id", "artist": "artist_id"}[item_type]
    status = "approved" if action == "approve" else "rejected"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {table} SET status = %s, reviewed_by = NULL WHERE {pk} = %s",
                (status, item_id),
            )
            conn.commit()
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
            # 与 viewer Artist 一致：SUM(popularity)，同一首歌在不同专辑出现多次则算多次；全库不按 decade 筛
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

            # Top tracks: 该艺人所有曲目按 popularity 降序，取每曲一行（多专辑则多行，取其一用 DISTINCT ON track_id 取 popularity 最高的一行）
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
            # 按 popularity 降序排，取前 10
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

            # Albums: 该艺人参与过的专辑
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
            # 按专辑名+年份去重，同一张专辑（同名同年版/豪华版等）只显示一张（保留第一条，已按 release_date DESC）
            seen_key = set()
            albums = []
            for a in albums_raw:
                key = (a["album_name"] or "", a["release_date"] or "")
                if key not in seen_key:
                    seen_key.add(key)
                    albums.append(a)

            # 歌手页背景：用最新一张专辑的封面（albums 已按 release_date DESC 排序）
            hero_cover = (albums[0]["album_image_url"] if albums else None) or (tracks[0]["album_image_url"] if tracks else None)
            # 自动人脸检测，使人脸落在 hero 裁切区内；无脸或失败则用默认居中
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


@app.route("/analyst/edit", methods=["GET", "POST"])
def analyst_edit():
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
            return redirect(url_for("analyst"))
        # Save
        if not track_id:
            flash("No track selected to save.")
            return redirect(url_for("analyst"))
        track_name = request.form.get("track_name", "").strip() or None
        duration_ms = request.form.get("duration_ms", "").strip()
        duration_ms = int(duration_ms) if duration_ms.isdigit() else None
        preview_url = request.form.get("preview_url", "").strip() or None
        release_date = request.form.get("release_date", "").strip() or None
        album_cover_url = request.form.get("album_cover_url", "").strip() or None
        genres = [g for g in request.form.getlist("genre") if g in BIG7]

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE tracks SET track_name=COALESCE(%s,track_name), duration_ms=COALESCE(%s,duration_ms), preview_url=COALESCE(%s,preview_url) WHERE track_id=%s",
                    (track_name, duration_ms, preview_url, track_id),
                )
                if release_date or album_cover_url is not None:
                    cur.execute(
                        """UPDATE albums SET release_date=COALESCE(%s,release_date), album_image_url=COALESCE(%s,album_image_url)
                           WHERE album_id = (SELECT album_id FROM album_tracks WHERE track_id=%s LIMIT 1)""",
                        (release_date if release_date else None, album_cover_url if album_cover_url else None, track_id),
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
                    SELECT t.track_id, t.track_name, t.duration_ms, t.preview_url,
                           ar.artist_id, ar.artist_name,
                           al.album_id, al.album_name, al.release_date, al.album_image_url
                    FROM tracks t
                    LEFT JOIN track_artist ta ON ta.track_id = t.track_id
                    LEFT JOIN artists ar ON ar.artist_id = ta.artist_id
                    LEFT JOIN album_tracks at ON at.track_id = t.track_id
                    LEFT JOIN albums al ON al.album_id = at.album_id
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
                        "artist_id": row[4],
                        "artist_name": row[5],
                        "album_id": row[6],
                        "album_name": row[7],
                        "release_date": str(row[8]) if row[8] else None,
                        "album_image_url": row[9],
                        "genre_names": genre_names,
                    }

    return render_template("analyst_edit.html", track=track, big7=BIG7)


if __name__ == "__main__":
    app.run(debug=True, port=5001)