# app/app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash
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
BIG7 = ["Pop", "Rock", "Hip-Hop", "R&B", "Jazz", "Classical", "Electronic"]

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
# VIEWER  (FIXED VERSION)
# -------------------------
@app.route("/viewer", methods=["GET"])
def viewer():
    """
    Key fixes:
    - chart in URL drives data (song/artist/album)
    - decade in URL truly filters by year
    - Big7 genres map correctly (no empty page)
    - rows remain 7 columns for songs
    """

    # ---- READ URL STATE ----
    chart = request.args.get("chart", "song")   # song | artist | album
    decade = request.args.get("decade", "All")
    genre = request.args.get("genre", "")
    q = request.args.get("q", "").strip()

    # normalize decade
    if decade.lower() == "all":
        decade = "All"

    where = ["t.status='approved'"]
    params = []

    # ---- DECADE FILTER ----
    if decade != "All":
        try:
            start_year = int(decade.rstrip("s"))
            where.append(
                """
                a.release_date IS NOT NULL
                AND EXTRACT(YEAR FROM a.release_date) >= %s
                AND EXTRACT(YEAR FROM a.release_date) < %s
                """
            )
            params.extend([start_year, start_year + 10])
        except Exception:
            pass

    # ---- BIG7 GENRE FILTER (CRITICAL FIX) ----
    if genre and genre in BIG7:
        where.append("""
        EXISTS (
          SELECT 1
          FROM track_artist ta2
          JOIN artist_genres ag2
            ON ag2.artist_id = ta2.artist_id
          WHERE ta2.track_id = t.track_id
            AND (
              (%s='Pop'        AND lower(ag2.genre_name) ~ '(pop|indie|k-pop|mandopop|synth)') OR
              (%s='Rock'       AND lower(ag2.genre_name) ~ '(rock|metal|punk|grunge|alternative)') OR
              (%s='Hip-Hop'    AND lower(ag2.genre_name) ~ '(hip hop|hip-hop|rap|trap)') OR
              (%s='R&B'        AND lower(ag2.genre_name) ~ '(r&b|rnb|soul|funk)') OR
              (%s='Jazz'       AND lower(ag2.genre_name) ~ '(jazz|swing|bebop)') OR
              (%s='Classical'  AND lower(ag2.genre_name) ~ '(classical|opera|symphony|piano|violin|orchestral)') OR
              (%s='Electronic' AND lower(ag2.genre_name) ~ '(edm|house|techno|trance|dubstep|electronic|electro)')
            )
        )
        """)
        params.extend([genre] * 7)

    # ---- SEARCH ----
    if q:
        where.append("""
        (lower(t.track_name) LIKE lower(%s)
         OR lower(ar.artist_name) LIKE lower(%s)
         OR lower(a.album_name) LIKE lower(%s))
        """)
        like = f"%{q}%"
        params.extend([like, like, like])

    # =========================
    # SONG TOP CHART (DEFAULT)
    # =========================
    if chart == "song":
        sql = f"""
        SELECT
          t.track_id,
          t.track_name,
          ar.artist_name,
          a.album_name,
          t.popularity,
          a.album_image_url,
          t.preview_url
        FROM tracks t
        JOIN album_tracks at ON at.track_id=t.track_id
        JOIN albums a ON a.album_id=at.album_id
        JOIN track_artist ta ON ta.track_id=t.track_id
        JOIN artists ar ON ar.artist_id=ta.artist_id
        WHERE {" AND ".join(where)}
        ORDER BY t.popularity DESC NULLS LAST
        LIMIT 50;
        """

    # =========================
    # ARTIST TOP CHART
    # =========================
    elif chart == "artist":
        sql = f"""
        SELECT
          ar.artist_id,
          ar.artist_name,
          COUNT(DISTINCT t.track_id) AS track_cnt,
          MAX(t.popularity) AS max_pop,
          MIN(a.album_image_url) AS any_cover,
          NULL AS preview_url
        FROM tracks t
        JOIN album_tracks at ON at.track_id=t.track_id
        JOIN albums a ON a.album_id=at.album_id
        JOIN track_artist ta ON ta.track_id=t.track_id
        JOIN artists ar ON ar.artist_id=ta.artist_id
        WHERE {" AND ".join(where)}
        GROUP BY ar.artist_id, ar.artist_name
        ORDER BY max_pop DESC NULLS LAST
        LIMIT 50;
        """

    # =========================
    # ALBUM TOP CHART
    # =========================
    else:  # album
        sql = f"""
        SELECT
          a.album_id,
          a.album_name,
          ar.artist_name,
          MAX(t.popularity) AS max_pop,
          a.album_image_url,
          NULL AS preview_url
        FROM tracks t
        JOIN album_tracks at ON at.track_id=t.track_id
        JOIN albums a ON a.album_id=at.album_id
        JOIN track_artist ta ON ta.track_id=t.track_id
        JOIN artists ar ON ar.artist_id=ta.artist_id
        WHERE {" AND ".join(where)}
        GROUP BY a.album_id, a.album_name, ar.artist_name, a.album_image_url
        ORDER BY max_pop DESC NULLS LAST
        LIMIT 50;
        """

    decades = ["All", "1980s", "1990s", "2000s", "2010s", "2020s"]
    genres = list(BIG7)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

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