# Track-level release date coverage comparison

## 1. CSV row-level (source data, one track per row)

| Metric | Value |
|--------|--------|
| Total rows (tracks) | 10,000 |
| Rows with Album Release Date parseable as date | 1,260 |
| Missing / unparseable rows | 8,740 |
| **Missing rate** | **87.40%** |
| Valid rate | 12.60% |

Note: "Valid" here means `pd.to_datetime(..., errors="coerce")` parses to a non-NaT value. Only about 2 rows (0.02%) are literally NaN/empty in the raw data; most rows become NaT after parsing, likely due to format or dtype.

---

## 2. DB track-level (release_date via album)

| Metric | Value |
|--------|--------|
| Total tracks | 9,952 |
| Tracks whose album has release_date | 8,653 |
| Tracks whose album has no release_date | 1,299 |
| **Missing rate** | **13.05%** |
| Valid rate | 86.95% |

Note: Tracks do not have a release_date column; "track has release date" is determined via album_tracks → albums.release_date.

---

## 3. Summary

| Level | Missing rate | Valid rate |
|-------|--------------|------------|
| CSV row-level (parseable as date) | 87.40% | 12.60% |
| DB track-level (album has release_date) | 13.05% | 86.95% |

Conclusion: After ETL, grouping by album and keeping one row per album with a date (when available) raises **track-level** release date coverage (via album) from ~12.6% to **86.95%** and reduces the missing rate from 87.4% to 13.05%.

---

## 4. NULL albums and source data validation

**1) Number of tracks on NULL-album (DB):**

```sql
SELECT COUNT(DISTINCT t.track_id) AS tracks_on_null_albums
FROM tracks t
JOIN album_tracks at ON at.track_id = t.track_id
JOIN albums al ON al.album_id = at.album_id
WHERE al.release_date IS NULL;
```

Result: **1,299** tracks lie on albums with release_date NULL.

**2) Do those NULL albums have "at least one row with a date" in the source?**

- In Python, using the same logic as ETL: group by `album_id_norm` and consider an album as having a date if at least one row has `Album Release Date` parseable as a date.
- If the CSV column has leading/trailing quotes (e.g. `'2009'`, `'2003-01-14'`), raw `pd.to_datetime` yields NaT; only ~1,260 rows parse at row level, corresponding to ~971 albums with a date.
- ETL already: (1) builds `release_date` from parsed `_rd`, avoiding double-parsing of year-only values; (2) strips leading/trailing quotes on the date column before parsing (e.g. `.str.strip().str.strip("'\"")`).
- Conclusion: Of the ~1,012 NULL albums, most have **no** parseable date row in the source (or the column is quote-wrapped and parsing fails). Only ~41 can be considered "strictly all empty" in the source. If more rows become parseable after stripping quotes in the CSV, re-running ETL would lower the NULL rate further.
