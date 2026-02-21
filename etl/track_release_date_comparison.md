# Track 层面 release date 缺失度对比

## 1. CSV 行级（源数据，每行一条 track）

| 指标 | 数值 |
|------|------|
| 总行数 (tracks) | 10,000 |
| 有 Album Release Date 且能解析为日期的行数 | 1,260 |
| 缺失/无法解析的行数 | 8,740 |
| **缺失率** | **87.40%** |
| 有效率 | 12.60% |

说明：此处“有效”指 `pd.to_datetime(..., errors="coerce")` 能解析为非 NaT。原始仅 NaN/空字符串约 2 行（0.02%），但多数行在解析后为 NaT，可能与格式或 dtype 有关。

---

## 2. DB Track 层面（通过专辑关联 release_date）

| 指标 | 数值 |
|------|------|
| 总 tracks 数 | 9,952 |
| 其专辑有 release_date 的 tracks 数 | 8,653 |
| 其专辑无 release_date 的 tracks 数 | 1,299 |
| **缺失率** | **13.05%** |
| 有效率 | 86.95% |

说明：track 无单独 release_date 列，通过 album_tracks → albums.release_date 判断“该 track 是否有发行日”。

---

## 3. 对比小结

| 维度 | 缺失率 | 有效率 |
|------|--------|--------|
| CSV 行级（可解析为日期） | 87.40% | 12.60% |
| DB track 级（其专辑有 release_date） | 13.05% | 86.95% |

结论：ETL 后按专辑聚合并优先保留“每个专辑内一条有日期的行”，使 **track 层面**（通过所属专辑）的 release date 覆盖率从约 12.6% 提升到约 **86.95%**，缺失度从 87.4% 降到 13.05%。

---

## 4. NULL 专辑与源数据验证

**1）NULL 专辑关联的 track 数（DB）：**

```sql
SELECT COUNT(DISTINCT t.track_id) AS tracks_on_null_albums
FROM tracks t
JOIN album_tracks at ON at.track_id = t.track_id
JOIN albums al ON al.album_id = at.album_id
WHERE al.release_date IS NULL;
```

结果：**1,299** 条 track 落在 release_date 为 NULL 的专辑上。

**2）这些 NULL 专辑在源数据里是否“有任意一行有日期”：**

- 在 Python 里用与 ETL 相同逻辑：按 `album_id_norm` 聚合并用「至少一行 `Album Release Date` 可解析为日期」判断。
- 若 CSV 中该列带首尾引号（如 `'2009'`、`'2003-01-14'`），直接 `pd.to_datetime` 会得到 NaT，行级仅约 1,260 行可解析，对应约 971 张专辑有日期。
- ETL 已做：① 用已解析的 `_rd` 生成 `release_date`，不再对「仅年份」等值二次解析；② 解析前对日期列做 `.str.strip().str.strip("'\"")` 去掉首尾引号。
- 结论：当前约 1,012 张 NULL 专辑中，绝大多数在源数据里**没有**可解析的日期行（或整列被引号包裹导致解析失败）；仅约 41 张在「严格全为空」意义下可视为源里就无日期。若 CSV 去引号后能解析更多行，重跑 ETL 后 NULL 率会再降。
