-- 已有库：把 track_genres 从「一 track 一行」改为「一 track 多 genre」
-- 执行：psql -h localhost -d musicbox -f db/migrate_track_genres_multi.sql
ALTER TABLE track_genres DROP CONSTRAINT IF EXISTS track_genres_pkey;
ALTER TABLE track_genres ADD PRIMARY KEY (track_id, genre_name);
