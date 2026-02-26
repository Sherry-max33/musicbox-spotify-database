# 🎧 MusicBox — Spotify Analytics Database & Web Application

> **A production-ready full-stack data product that transforms raw Spotify metadata into an interactive analytics platform for music exploration and data-driven business decision-making.**

👉 **Live Demo:** [https://musicbox-spotify-database.onrender.com/viewer](https://musicbox-spotify-database.onrender.com/viewer)  
*(Note: Hosted on a free tier; the initial load may require ~30 seconds for server wake-up.)*

---

## 🌟 Project Overview
MusicBox is an end-to-end data application demonstrating how structured data infrastructure powers analytics for streaming platforms and music labels. It features a custom ETL pipeline that cleans and ingests ~10,000 Spotify tracks from raw CSV format into a normalized PostgreSQL database. 

The system integrates data engineering, relational modeling, and backend services into a deployable cloud application designed to simulate real-world digital product environments.

---

## 🏗️ System Architecture
**Data Source (CSV)** → **ETL Processing (Python/Pandas)** → **PostgreSQL (Supabase)** → **Flask Backend (SQL/psycopg)** → **Jinja/HTML Web UI** → **Cloud Deployment (Render + Supabase)**

---

## 🗃️ Database Design & Metric Logic
The relational schema is designed using database normalization principles (3NF) to support scalable analytics queries and flexible aggregations.

### 📊 Entity-Relationship Diagram (ERD)
![MusicBox ERD](image_demo/supabase-ERD.png)  
*Visual representation of the normalized schema modeling Tracks, Artists, Albums, and their complex relationships.*

### 1. Categorical Mapping (Genre Consolidation)
* **The Challenge:** Raw Spotify genres are highly fragmented (e.g., "trap latino," "latin pop").
* **The Solution:** Engineered a mapping pipeline that collapses 500+ granular sub-genres into a structured **"Big-8" Genre System**.

### 2. Popularity Metric Design
* **Scoring Logic:** $Score = \sum(Popularity)$.
* **E-commerce Analogy:** Similar to **GMV** in e-commerce, this metric reflects "Scale of Influence." `DISTINCT ON (track_id)` ensures each track is counted once per entity.

---

## 📈 Business Intelligence & Actionable Insights
Leveraging a 10-year background in e-commerce operations, this project extracts commercial value from audio data through three core analytical lenses:

### 1. User Profiling & Personalization (The "Style Map")
* **Acoustic Fingerprinting:** The Radar Chart quantifies key audio features like danceability, energy, and valence.
* **Business Value:** This allows platforms to define "User Taste Profiles," enabling targeted high-BPM playlists to increase Click-Through Rates (CTR).

### 2. Trend Forecasting & Market Intelligence (Style Evolution)
* **Genre Mix Over Time:** Visualizes the shifts in genre popularity and audio characteristics across decades.
* **Business Value:** By identifying "rising" genres, labels can make data-driven decisions on talent scouting.

### 3. Retention & Churn Prediction
* **Audio Feature Evolution:** Tracks the temporal evolution of listening habits to support re-engagement campaigns.

---

## 🛠️ Tech Stack

### Backend
* **Python** | **Flask** | **psycopg** (PostgreSQL driver)

### Database
* **PostgreSQL** (Supabase)

### Frontend
* **HTML + CSS** | **Jinja** Templating | **JavaScript** (Audio Playback)

### Data Processing / ETL
* **Python** (Pandas, NumPy)

### Deployment
* **Render** (Gunicorn) | **Supabase** (Managed DB)

---

## 🧠 Key Features
* **🔍 Search & Exploration:** Cross-entity search and detailed metadata relationships.
* **📊 Audio & Genre Analytics:** Dynamic charts with decade filters powered by aggregate SQL.
* **👥 Role-Based Views:** Viewer interface vs. Analyst/Admin console with **Undo** functionality.
* **⚡ Optimized Queries:** Hand-tuned SQL joins and **window functions** for real-time ranking.

---

## 💼 Skills Demonstrated
* **Data Engineering:** 3NF Schema design, ETL pipeline development.
* **Backend Development:** Flask API routing and PostgreSQL integration.
* **Product Thinking:** Role-based workflow design and business metric definition.
* **Cloud Deployment:** Production-grade hosting and environment secrets management.