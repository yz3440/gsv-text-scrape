from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlite3
from typing import Optional
import uvicorn
import os
import logging
import dotenv

dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the database file
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsv.db")
GOOGLE_MAP_API_KEY = os.getenv("GOOGLE_MAP_API_KEY")

# Log the database path on startup
logger.info(f"Database path: {DB_PATH}")
logger.info(f"Database exists: {os.path.exists(DB_PATH)}")


def get_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise


@app.get("/api/streetview-url/{pano_id}")
async def get_streetview_url(pano_id: str):
    """Generate a Google Street View URL for a given panorama ID."""
    conn = None
    try:
        logger.info(f"Looking up panorama ID: {pano_id}")
        conn = get_db()
        cursor = conn.cursor()

        # First check if the table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='search_panoramas'"
        )
        if not cursor.fetchone():
            logger.error("search_panoramas table does not exist")
            raise HTTPException(status_code=500, detail="Database table not found")

        # Then check for the panorama
        cursor.execute("SELECT * FROM search_panoramas WHERE pano_id = ?", (pano_id,))
        result = cursor.fetchone()
        if not result:
            logger.error(f"Panorama ID {pano_id} not found in database")
            raise HTTPException(
                status_code=404, detail=f"Panorama ID {pano_id} not found in database"
            )

        # Log the actual data we found
        logger.info(f"Found panorama data: {dict(result)}")

        url = f"https://www.google.com/maps/embed/v1/streetview?key={GOOGLE_MAP_API_KEY}&pano={pano_id}&heading=0&pitch=0&fov=90"
        logger.info(f"Generated Street View URL for pano_id: {pano_id}")
        return {"url": url}
    except Exception as e:
        logger.error(f"Error generating Street View URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/panoramas")
async def get_panoramas(
    page: int = 1, page_size: int = 50, search: Optional[str] = None
):
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Get total count
        count_query = "SELECT COUNT(*) as total FROM search_panoramas"
        if search:
            count_query += " WHERE pano_id LIKE ?"
            cursor.execute(count_query, (f"%{search}%",))
        else:
            cursor.execute(count_query)
        total = cursor.fetchone()["total"]

        # Get paginated data
        offset = (page - 1) * page_size
        query = "SELECT * FROM search_panoramas"
        if search:
            query += " WHERE pano_id LIKE ?"
            query += " ORDER BY pano_id LIMIT ? OFFSET ?"
            cursor.execute(query, (f"%{search}%", page_size, offset))
        else:
            query += " ORDER BY pano_id LIMIT ? OFFSET ?"
            cursor.execute(query, (page_size, offset))

        rows = cursor.fetchall()

        # Convert rows to list of dicts
        panoramas = [dict(row) for row in rows]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "data": panoramas,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/panorama/{pano_id}")
async def get_panorama(pano_id: str):
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM search_panoramas WHERE pano_id = ?", (pano_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Panorama not found")

        return dict(row)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
