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
"""
import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import psycopg2
from dotenv import load_dotenv

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


# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("viewer"))


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
          SELECT artist_id, artist_name, track_name, preview_url, popularity,
                 ROW_NUMBER() OVER (PARTITION BY artist_id ORDER BY popularity DESC NULLS LAST) AS rn
          FROM decade_tracks
        ),
        artist_score AS (
          SELECT artist_id, SUM(popularity)::numeric AS score FROM decade_tracks GROUP BY artist_id
        )
        SELECT r.artist_name, r.track_name, r.preview_url
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
          SELECT a.album_id, ar.artist_name, SUM(t.popularity) AS artist_pop
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
          SELECT DISTINCT ON (album_id) album_id, artist_name
          FROM album_artist_pop
          ORDER BY album_id, artist_pop DESC NULLS LAST
        )
        SELECT tot.album_name, tot.album_image_url, ba.artist_name
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
    with get_conn() as conn:
        with conn.cursor() as cur:
            if chart_type == "songs":
                sql = """
                SELECT *
                FROM (
                  SELECT DISTINCT ON (t.track_id)
                    t.track_id, t.track_name, t.popularity, t.preview_url,
                    ar.artist_name, al.album_name, al.album_image_url
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
                columns = [
                    "track_id", "track_name", "popularity", "preview_url",
                    "artist_name", "album_name", "album_image_url",
                ]
                items = [
                    {
                        "rank": i + 1,
                        "track_name": r[1],
                        "artist_name": r[4],
                        "album_name": r[5],
                        "album_image_url": r[6],
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
                SELECT o.artist_name, o.track_name, o.preview_url, o.album_image_url
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
                        "artist_name": r[0],
                        "top_track_name": r[1],
                        "preview_url": r[2],
                        "album_image_url": r[3],
                    }
                    for i, r in enumerate(rows)
                ]
            else:
                # albums
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
                  SELECT a.album_id, ar.artist_name, SUM(t.popularity) AS artist_pop
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
                  SELECT DISTINCT ON (album_id) album_id, artist_name
                  FROM album_artist_pop
                  ORDER BY album_id, artist_pop DESC NULLS LAST
                )
                SELECT tot.album_name, tot.album_image_url, ba.artist_name
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
                        "album_name": r[0],
                        "album_image_url": r[1],
                        "artist_name": r[2],
                    }
                    for i, r in enumerate(rows)
                ]
    return jsonify({"items": items, "genre": genre, "decade": decade_param, "type": chart_type})


# -------------------------
# Analyst (UNCHANGED)
# -------------------------
@app.route("/analyst", methods=["GET"])
def analyst():
    genre = request.args.get("genre", "").strip()
    feature = request.args.get("feature", "danceability").strip()

    if feature not in FEATURE_TABS:
        feature = "danceability"

    if genre and genre not in BIG7:
        genre = ""

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

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(genres_sql, (BIG7,))
            genres = [r[0] for r in cur.fetchall()] or BIG7[:]

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

            cur.execute(mix_sql, (BIG7,))
            mix_rows = cur.fetchall()

            mix_by_decade = {}
            for decade_int, g, cnt in mix_rows:
                if decade_int is None or g is None:
                    continue
                dlab = decade_label(decade_int)
                mix_by_decade.setdefault(dlab, {k: 0 for k in BIG7})
                if g in mix_by_decade[dlab]:
                    mix_by_decade[dlab][g] += int(cnt)

            genre_mix = []
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

            cur.execute(trend_sql, (BIG7, genre, genre))
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

    return render_template(
        "analyst.html",
        genres=genres,
        genre=genre,
        active_feature=feature,
        summary=summary,
        radar_data=radar_data,
        genre_mix=genre_mix,
        trend_data=trend_data,
    )


# -------------------------
# Admin (unchanged)
# -------------------------
@app.route("/admin/login", methods=["GET"])
def admin_login():
    return render_template("admin_login.html")


@app.route("/admin", methods=["GET"])
def admin():
    return render_template("admin_panel.html")


@app.route("/artist", methods=["GET"])
def artist():
    return render_template("artist.html")


@app.route("/analyst/edit", methods=["GET"])
def analyst_edit():
    return render_template("analyst_edit.html")


if __name__ == "__main__":
    app.run(debug=True)