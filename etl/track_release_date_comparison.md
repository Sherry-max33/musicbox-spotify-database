# Track-level release date coverage comparison

This doc describes release date coverage at CSV vs DB level. The ETL **uses regex-based normalization** (`normalize_release_date_str` in `etl_load.py`) before parsing: strip quotes, normalize dashes/slashes, keep only digits and hyphen. So more rows become parseable than raw `pd.to_datetime` on the CSV column; the numbers in section 1 are **without** that normalization (raw CSV); section 2 reflects the current ETL output **with** normalization.

## 1. CSV row-level (source data, one track per row) — *raw, no ETL normalization*

| Metric | Value |
|--------|--------|
| Total rows (tracks) | 10,000 |
| Rows with Album Release Date parseable as date | 1,260 |
| Missing / unparseable rows | 8,740 |
| **Missing rate** | **87.40%** |
| Valid rate | 12.60% |

Note: "Valid" here means raw `pd.to_datetime(..., errors="coerce")` on the CSV column parses to non-NaT. Many rows fail due to quotes, dashes, or format. **After ETL’s regex normalization**, more rows parse (see `normalize_release_date_str` in `etl_load.py`).

---

## 2. DB track-level (release_date via album) — *current DB after latest ETL*

| Metric | Value |
|--------|--------|
| Total tracks | 9,952 |
| Tracks whose album has release_date | 9,948 |
| Tracks whose album has no release_date | 4 |
| **Missing rate** | **~0.04%** |
| Valid rate | ~99.96% |

Note: Tracks do not have a release_date column; "track has release date" is determined via album_tracks → albums.release_date. Only 3 albums have NULL release_date, affecting 4 tracks total.

---

## 3. Summary

| Level | Missing rate | Valid rate |
|-------|--------------|------------|
| CSV row-level (parseable as date, raw) | 87.40% | 12.60% |
| DB track-level (after latest ETL) | **~0.04%** | **~99.96%** |

Conclusion: After ETL (including **regex normalization** of the date column, then grouping by album and keeping one row per album with a date when available), **track-level** release date coverage goes from ~12.6% (raw CSV) to **~99.96%** (current DB). Missing rate drops from 87.4% to **~0.04%** (4 tracks on 3 NULL albums).

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

Result: **4** tracks lie on albums with release_date NULL (3 albums). Missing rate ≈ 0.04%.

**2) Do those NULL albums have "at least one row with a date" in the source?**

- In Python, using the same logic as ETL: group by `album_id_norm` and consider an album as having a date if at least one row has `Album Release Date` parseable after **the same regex normalization** (`normalize_release_date_str` → `parse_release_date_to_year`).
- **After regex normalization**, the current ETL achieves ~99.96% track-level coverage (only 4 tracks on 3 albums without a date). Earlier runs or raw-CSV comparisons showed higher NULL counts; the numbers in section 1 (87% missing) are for **raw** CSV parsing without normalization.
- ETL: (1) uses `normalize_release_date_str()` (regex) then `parse_release_date_to_year()`; (2) builds `release_date` from parsed `_rd`; (3) one row per album, keeping a row with a date when available.
- Conclusion: With the current pipeline, only 3 albums (4 tracks) have no release_date. If you improve normalization or fix more formats, re-running ETL could reduce this further.
