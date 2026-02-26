-- db/migrate_admin_indexes.sql
-- Speed up admin_panel Content Review queries on large datasets.
--
-- Run on your Postgres (local or Supabase):
--   psql <connection-string> -v ON_ERROR_STOP=1 -f db/migrate_admin_indexes.sql

CREATE INDEX IF NOT EXISTS idx_tracks_status_reviewed_at
  ON tracks (status, reviewed_at DESC);

CREATE INDEX IF NOT EXISTS idx_albums_status_reviewed_at
  ON albums (status, reviewed_at DESC);

CREATE INDEX IF NOT EXISTS idx_artists_status_reviewed_at
  ON artists (status, reviewed_at DESC);

CREATE INDEX IF NOT EXISTS idx_tracks_status
  ON tracks (status);

CREATE INDEX IF NOT EXISTS idx_albums_status
  ON albums (status);

CREATE INDEX IF NOT EXISTS idx_artists_status
  ON artists (status);

