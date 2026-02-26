-- db/schema.sql
DROP TABLE IF EXISTS album_tracks CASCADE;
DROP TABLE IF EXISTS track_artist CASCADE;
DROP TABLE IF EXISTS track_genres CASCADE;
DROP TABLE IF EXISTS artist_primary_genre CASCADE;
DROP TABLE IF EXISTS artist_genres CASCADE;
DROP TABLE IF EXISTS audio_features CASCADE;
DROP TABLE IF EXISTS tracks CASCADE;
DROP TABLE IF EXISTS albums CASCADE;
DROP TABLE IF EXISTS artists CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
  user_id BIGSERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255),
  role VARCHAR(20) NOT NULL CHECK (role IN ('analyst','admin')),
  is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE artists (
  artist_id VARCHAR(64) PRIMARY KEY,
  artist_name VARCHAR(300) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'approved' CHECK (status IN ('pending','approved','rejected')),
  added_at DATE,
  submitted_by BIGINT REFERENCES users(user_id),
  reviewed_by BIGINT REFERENCES users(user_id),
  reviewed_at TIMESTAMP
);

CREATE TABLE albums (
  album_id VARCHAR(64) PRIMARY KEY,
  album_name VARCHAR(400) NOT NULL,
  release_date DATE,
  album_image_url TEXT,
  status VARCHAR(20) NOT NULL DEFAULT 'approved' CHECK (status IN ('pending','approved','rejected')),
  added_at DATE,
  submitted_by BIGINT REFERENCES users(user_id),
  reviewed_by BIGINT REFERENCES users(user_id),
  reviewed_at TIMESTAMP
);

CREATE TABLE tracks (
  track_id VARCHAR(64) PRIMARY KEY,
  track_name VARCHAR(400) NOT NULL,
  popularity SMALLINT,
  duration_ms INT,
  explicit BOOLEAN,
  preview_url TEXT,
  status VARCHAR(20) NOT NULL DEFAULT 'approved' CHECK (status IN ('pending','approved','rejected')),
  added_at DATE,
  submitted_by BIGINT REFERENCES users(user_id),
  reviewed_by BIGINT REFERENCES users(user_id),
  reviewed_at TIMESTAMP
);

CREATE TABLE track_artist (
  track_id VARCHAR(64) NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
  artist_id VARCHAR(64) NOT NULL REFERENCES artists(artist_id) ON DELETE CASCADE,
  PRIMARY KEY (track_id, artist_id)
);

CREATE TABLE album_tracks (
  album_id VARCHAR(64) NOT NULL REFERENCES albums(album_id) ON DELETE CASCADE,
  track_id VARCHAR(64) NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
  disc_number SMALLINT,
  track_number SMALLINT,
  PRIMARY KEY (album_id, track_id)
);

CREATE TABLE artist_genres (
  artist_id VARCHAR(64) NOT NULL REFERENCES artists(artist_id) ON DELETE CASCADE,
  genre_name VARCHAR(120) NOT NULL,
  PRIMARY KEY (artist_id, genre_name)
);

-- One row per artist: primary Big-8 genre = primary genre of that artist's highest-popularity track that has track_genres (aligned with song/album)
CREATE TABLE artist_primary_genre (
  artist_id VARCHAR(64) PRIMARY KEY REFERENCES artists(artist_id) ON DELETE CASCADE,
  genre_name VARCHAR(120) NOT NULL
);

-- Multiple rows per track: one track can have multiple Big-8 genres (multi-select on edit page)
CREATE TABLE track_genres (
  track_id VARCHAR(64) NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
  genre_name VARCHAR(120) NOT NULL,
  PRIMARY KEY (track_id, genre_name)
);

CREATE TABLE audio_features (
  track_id VARCHAR(64) PRIMARY KEY REFERENCES tracks(track_id) ON DELETE CASCADE,
  danceability DECIMAL(5,4),
  energy DECIMAL(5,4),
  valence DECIMAL(5,4),
  acousticness DECIMAL(7,6),
  instrumentalness DECIMAL(10,9),
  liveness DECIMAL(5,4),
  speechiness DECIMAL(5,4),
  tempo DECIMAL(7,3),
  key SMALLINT,
  mode SMALLINT,
  loudness DECIMAL(7,3),
  time_signature SMALLINT
);

-- seed users for demo (John Doe, Jane Doe, analyst, Alex Smith)
INSERT INTO users (email, password_hash, role, is_active)
VALUES
('admin@musicbox.local', 'demo', 'admin', TRUE),
('analyst@musicbox.local', 'demo', 'analyst', TRUE),
('jane.doe@musicbox.local', 'demo', 'analyst', TRUE),
('alex.smith@musicbox.local', 'demo', 'analyst', TRUE);
-- MusicBox database schema

-- TODO: add your tables here
