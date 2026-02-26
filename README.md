🎧 MusicBox — Spotify Analytics Database & Web Application

A production-style full-stack data product that transforms raw Spotify data into an interactive analytics platform for music exploration and decision-making.

👉 Live Demo: https://musicbox-spotify-database.onrender.com/viewer

🌟 Project Overview

MusicBox is an end-to-end data application that demonstrates how structured data infrastructure can power analytics experiences similar to those used by streaming platforms, music labels, and digital marketers.

The system integrates data engineering, database design, backend services, and a user-facing web interface into a deployable cloud application.

It is built on top of a curated ~10K-track Spotify dataset (CSV), ingested via a custom ETL pipeline into PostgreSQL.

💡 Motivation

Modern digital products rely on robust data pipelines and analytics layers to extract insights from large-scale behavioral and content data.

This project was built to simulate a real-world analytics platform that enables users to:

- Explore music metadata
- Analyze audio characteristics
- Perform search and discovery
- Support data-driven decisions

🧠 Key Features

🔍 Search & Exploration

- Search across tracks, artists, and albums
- Browse artist and album detail pages
- View rich track/album metadata and relationships

📊 Audio & Genre Analytics

- Spotify audio features (danceability, energy, valence, etc.)
- Big‑8 genre system with decade filters and “Explore by Genres” charts
- Top charts and trend-style views powered by aggregate SQL over popularity and time

👥 Role-Based Views

- Viewer interface for casual exploration
- Analyst/Admin console for editing metadata and running content review workflows
- Review queue + decision history with undo, mirroring real analytics/review tools

🎵 Media Integration

- Album artwork display
- In-browser audio preview playback

⚡ Performance-Optimized Queries

- Normalized relational schema designed for fast lookups
- Hand-tuned SQL joins and window functions for top charts and genre breakdowns

🏗️ System Architecture

Data Sources (Spotify CSV) → ETL Processing (Python/Pandas) → PostgreSQL (Supabase) → Flask Backend (SQL/psycopg) → Jinja/HTML Web UI  
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　↓  
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 Cloud Deployment (Render + Supabase)

🗃️ Database Design

Normalized relational schema modeling core music entities:

- Tracks
- Artists
- Albums
- Audio Features
- Track–Artist relationships
- Track–Album relationships
- Genre mapping tables (fine-grained → Big‑8 categories)

Designed using database normalization principles to support scalable analytics queries and flexible aggregations.

🛠️ Tech Stack

**Backend**

- Python
- Flask
- psycopg (PostgreSQL driver) + raw SQL queries

**Database**

- PostgreSQL (Supabase)

**Frontend**

- HTML + CSS
- Jinja server-side templating
- Lightweight JavaScript for interactivity and audio playback

**Data Processing / ETL**

- Python (Pandas, NumPy)
- Batch ETL script that cleans and loads the Spotify CSV into PostgreSQL

**Deployment**

- Render (Flask app hosting with Gunicorn)
- Supabase (managed PostgreSQL)

🚀 Deployment

The application is deployed as a public cloud service:

- Backend hosted on Render
- Database hosted on Supabase
- Accessible via browser without local setup

👉 Live Viewer:
https://musicbox-spotify-database.onrender.com/viewer

🎯 Use Cases

MusicBox demonstrates capabilities relevant to:

- Data Analytics Platforms
- Business Intelligence Tools
- Digital Media Analytics
- Product Data Infrastructure
- Data-Driven Decision Systems

💼 Skills Demonstrated

**Data Engineering**

- Data cleaning and ingestion from CSV
- Relational schema design for analytics
- SQL query design and optimization

**Backend Development**

- Flask routing and view design
- PostgreSQL integration via psycopg
- JSON APIs for charting and interactive UI components

**Product & UX Thinking**

- Role-based interfaces (viewer vs. analyst/admin)
- Review workflows (queues, decisions, undo)
- Analytics-focused exploration flows (genres, decades, top charts)

**Cloud Deployment**

- Production-style hosting on Render (Gunicorn)
- Managed database integration with Supabase
- Environment-based configuration and secrets management