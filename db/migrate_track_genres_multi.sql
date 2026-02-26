-- Existing DB: change track_genres from one row per track to multiple genres per track.
-- Run: psql -h localhost -d musicbox -f db/migrate_track_genres_multi.sql
ALTER TABLE track_genres DROP CONSTRAINT IF EXISTS track_genres_pkey;
ALTER TABLE track_genres ADD PRIMARY KEY (track_id, genre_name);
