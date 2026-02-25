-- Add reviewed_at for Content Review "Processed At" column.
-- Run once on existing DBs: psql -f db/migrations/add_reviewed_at.sql

ALTER TABLE tracks  ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP;
ALTER TABLE albums  ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP;
ALTER TABLE artists ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP;
