# Morocco Motorcycle Travel Recommendation System

AI-powered motorcycle trip recommendation engine for Morocco.  
Combines **structured data**, **vector search (FAISS)**, **GPX route parsing**, and **LLM generation (Ollama / Gemma2)** to produce personalised day-by-day itineraries.

---

## Architecture

```
┌──────────────┐   ┌─────────────┐   ┌─────────────┐
│  JSON KB     │   │ Markdown /  │   │  GPX files  │
│  (places,    │   │ travel text │   │  (tracks,   │
│  routes, …)  │   │             │   │  waypoints) │
└──────┬───────┘   └──────┬──────┘   └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  ▼
   ┌────────┐      ┌───────────┐      ┌───────────┐
   │ SQLite │      │ Chunking  │      │ GPX Parse │
   │  Store │      │ + Embed   │      │ + Enrich  │
   └───┬────┘      └─────┬─────┘      └─────┬─────┘
       │                  │                  │
       │                  ▼                  │
       │           ┌───────────┐             │
       │           │   FAISS   │             │
       │           │  Vectors  │             │
       │           └─────┬─────┘             │
       │                 │                   │
       ▼                 ▼                   ▼
   ┌──────────────────────────────────────────────┐
   │          Recommendation Engine               │
   │  filter → score → retrieve → assemble → LLM │
   └─────────────────────┬────────────────────────┘
                         │
                         ▼
                   ┌───────────┐
                   │  FastAPI  │
                   │  /recommend
                   └───────────┘
```

### Key Modules

| Layer | Path | Purpose |
|-------|------|---------|
| **API** | `app/api/routes.py` | FastAPI endpoints |
| **Schemas** | `app/schemas/` | Pydantic request/response models |
| **Domain** | `app/models/` | Place, Route, Stay, etc. |
| **Repos** | `app/repositories/` | SQLite query layer |
| **Services** | `app/services/` | Ingestion, GPX, engine, Ollama |
| **Vector** | `app/vector_store/` | FAISS + embeddings |
| **Config** | `app/core/` | Settings, DB init |

---

## Installation

### Prerequisites
- **Python 3.11+**
- **Ollama** installed and running locally → https://ollama.com

### 1. Clone & install Python dependencies

```bash
cd marocain_riders
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull the LLM model

```bash
ollama pull gemma2:9b
```

Verify it's available:

```bash
ollama list
```

### 3. Configure environment

Copy the example and adjust if needed:

```bash
cp .env.example .env
```

Default values work out of the box for local development.

---

## Running

### Step 1: Ingest data

Populate the SQLite database, chunk text, and build the FAISS index:

```bash
python -m scripts.ingest_all
```

Or use the API endpoints (after starting the server):

```bash
# POST http://localhost:8000/ingest/json
# POST http://localhost:8000/ingest/text
# POST http://localhost:8000/ingest/gpx
```

### Step 2: Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/ingest/json` | Ingest JSON knowledge base → SQLite |
| `POST` | `/ingest/text` | Ingest text/markdown → chunks → FAISS |
| `POST` | `/ingest/gpx` | Parse GPX files → vector enrichment |
| `POST` | `/recommend` | Full recommendation pipeline |

---

## Example `/recommend` Request

```json
{
  "user_profile": {
    "rider_experience": "intermediate",
    "bike_type": "adventure",
    "group_size": 2,
    "budget_per_day_eur": 60,
    "trip_duration_days": 7,
    "preferred_vibes": ["desert", "culture"],
    "sleep_preference": "mid-range",
    "daily_ride_km_tolerance": 200,
    "travel_month": "October",
    "country_of_origin": "France"
  },
  "additional_preferences": "We love gorge roads and want at least one night in the desert"
}
```

### Example Response Structure

```json
{
  "recommended_route": {
    "route_id": "RT001",
    "route_name": "Grand Atlas Circuit",
    "total_score": 0.87,
    "score_breakdown": {
      "difficulty": 0.75,
      "bike_suit": 1.0,
      "vibe": 0.85,
      "duration": 1.0,
      "fuel_safety": 0.8,
      "season": 1.0
    },
    "distance_km": 1450,
    "difficulty_level": "intermediate",
    "recommended_days": "7-10",
    "why_choose": "The complete Morocco south experience"
  },
  "alternative_routes": [ ... ],
  "day_plan": [
    {
      "day_number": 1,
      "start_place": "Marrakech",
      "end_place": "Ouarzazate",
      "distance_km": 160,
      "stops": [ ... ]
    }
  ],
  "food_stops": [ ... ],
  "stay_stops": [ ... ],
  "fuel_stops": [ ... ],
  "weather_warnings": [ ... ],
  "safety_notes": [ ... ],
  "llm_explanation": "Based on your profile as an intermediate adventure rider ...",
  "retrieved_context_summary": [ ... ]
}
```

---

## Data Sources

| Source | Location | Format |
|--------|----------|--------|
| Structured KB | `app/data/json/knowledge_base.json` | JSON |
| Travel descriptions | `app/data/docs/` | Markdown / text |
| GPX routes | `app/data/gpx/` | GPX 1.1 |

---

## Testing

```bash
pytest tests/ -v
```

---

## Project Structure

```
marocain_riders/
├── app/
│   ├── api/
│   │   ├── dependencies.py        # DI / singletons
│   │   └── routes.py              # FastAPI endpoints
│   ├── core/
│   │   ├── config.py              # Settings from .env
│   │   └── database.py            # SQLite setup
│   ├── data/
│   │   ├── json/knowledge_base.json
│   │   ├── docs/places_routes.md
│   │   └── gpx/High-Atlas-Traverse-v1.9.gpx
│   ├── models/
│   │   ├── domain.py              # Place, Route, Stay, …
│   │   └── gpx_models.py          # GPX data structures
│   ├── repositories/
│   │   ├── place_repo.py
│   │   ├── route_repo.py
│   │   └── entity_repos.py        # Segments, Stays, Restaurants, …
│   ├── schemas/
│   │   ├── request.py             # UserProfile, RecommendationRequest
│   │   └── response.py            # RecommendationResponse, DayPlan, …
│   ├── services/
│   │   ├── json_ingestion.py
│   │   ├── text_ingestion.py
│   │   ├── gpx_parser.py
│   │   ├── ollama_client.py
│   │   ├── prompt_builder.py
│   │   └── recommendation_engine.py
│   ├── utils/
│   │   └── geo.py                 # Haversine, midpoint
│   ├── vector_store/
│   │   └── faiss_store.py         # FAISS index + metadata
│   └── main.py                    # FastAPI app factory
├── scripts/
│   └── ingest_all.py              # CLI ingestion script
├── tests/
│   └── test_scoring.py
├── .env.example
├── .env
├── requirements.txt
└── README.md
```

---

## Future Improvements (TODOs)

- [ ] **React + Leaflet frontend** consuming the `/recommend` API
- [ ] **Map tile integration** rendering GPX tracks on interactive maps
- [ ] **Streaming LLM responses** via SSE for real-time itinerary display
- [ ] **Multi-route GPX overlay** linking each route to its GPX data
- [ ] **User authentication** and trip history storage
- [ ] **PDF upload endpoint** with automatic text extraction (PyMuPDF)
- [ ] **Caching layer** (Redis) for repeated recommendation queries
- [ ] **More granular budget scoring** using actual price ranges
- [ ] **Elevation profile charts** from GPX track data
- [ ] **Seasonal calendar API** returning best routes per month
- [ ] **Docker Compose** setup for Ollama + API deployment
- [ ] **Multi-language support** (French, Arabic, Spanish)
- [ ] **Feedback loop** — store user ratings to improve scoring weights

---

## License

Private project — not licensed for redistribution.
