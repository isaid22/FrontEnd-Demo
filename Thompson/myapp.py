from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from datetime import datetime, timezone

app = FastAPI(title="Recommender API", default_response_class=ORJSONResponse)

# ✅ Dev CORS: allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # or ["http://localhost:8080"] if you want to be stricter
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,    # keep False if using "*"
)

@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.get("/api/recommendation")
async def recommendation(user_id: str):
    # Keep the original 'items' shape for compatibility, but also add
    # the simple top-level fields the frontend expects: `message`, `url`,
    # and `recommendation` (arm id). This prevents client errors when the
    # UI tries to access `json.message` and `json.recommendation`.
    items = [
        {
            "id": "house-hamptons-1",
            "title": "Modern East Hampton Home",
            "image": "https://picsum.photos/seed/hamptons/640/360",
            "score": 0.91,
            "cta_url": "https://www.zillow.com/homedetails/18-E-Hollow-Rd-East-Hampton-NY-11937/32650917_zpid/"
        }
    ]

    first = items[0] if items else None
    message = f"<link>{first['title']}</link>" if first else "No recommendation available"
    url = first.get('cta_url') if first else ""
    recommendation = first.get('id') if first else None

    return {
        "user_id": user_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
        # Frontend-friendly top-level fields
        "message": message,
        "url": url,
        "recommendation": recommendation
    }
