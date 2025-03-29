from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
from typing import Optional, List
import uvicorn
import os
import logging
import dotenv
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util.gsv_url import get_google_streetview_props, get_google_streetview_embed_url

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

# Get the absolute path to the database file and static directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gsv.db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
GOOGLE_MAP_API_KEY = os.getenv("GOOGLE_MAP_API_KEY")

# Log the paths on startup
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Database path: {DB_PATH}")
logger.info(f"Static directory: {STATIC_DIR}")
logger.info(f"Database exists: {os.path.exists(DB_PATH)}")
logger.info(f"Static directory exists: {os.path.exists(STATIC_DIR)}")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def read_root():
    """Serve the index.html file."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/preview-db")
async def read_preview_db():
    """Serve the preview-db.html file."""
    file_path = os.path.join(STATIC_DIR, "preview-db.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="preview-db.html not found")
    return FileResponse(file_path)


@app.get("/preview-ocr")
async def read_preview_ocr():
    """Serve the preview-ocr.html file."""
    file_path = os.path.join(STATIC_DIR, "preview-ocr.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="preview-ocr.html not found")
    return FileResponse(file_path)


@app.get("/preview-db-ocr")
async def read_preview_db_ocr():
    """Serve the preview-db-ocr.html file."""
    file_path = os.path.join(STATIC_DIR, "preview-db-ocr.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="preview-db-ocr.html not found")
    return FileResponse(file_path)


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


@app.get("/api/ocr-search")
async def search_ocr(
    query: str,
    page: int = 1,
    page_size: int = 50,
    min_confidence: Optional[float] = None,
):
    """Search through OCR results and return matching entries with panorama information."""
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Base query to join OCR results with panorama information
        base_query = """
            SELECT 
                ocr.id,
                ocr.pano_id,
                ocr.text,
                ocr.confidence,
                ocr.yaw,
                ocr.pitch,
                ocr.width,
                ocr.height,
                ocr.engine,
                sp.lat,
                sp.lon,
                sp.heading,
                sp.pitch as panorama_pitch,
                sp.roll,
                sp.date,
                sp.copyright
            FROM ocr_result ocr
            JOIN search_panoramas sp ON ocr.pano_id = sp.pano_id
            WHERE ocr.text LIKE ?
        """

        # Add confidence filter if specified
        if min_confidence is not None:
            base_query += " AND ocr.confidence >= ?"
            params = [f"%{query}%", min_confidence]
        else:
            params = [f"%{query}%"]

        # Get total count
        count_query = f"SELECT COUNT(*) as total FROM ({base_query})"
        cursor.execute(count_query, params)
        total = cursor.fetchone()["total"]

        # Get paginated results
        offset = (page - 1) * page_size
        query = base_query + " ORDER BY ocr.confidence DESC LIMIT ? OFFSET ?"
        cursor.execute(query, params + [page_size, offset])

        rows = cursor.fetchall()
        results = [dict(row) for row in rows]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "data": results,
        }

    except Exception as e:
        logger.error(f"Error searching OCR results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/ocr-streetview-url/{pano_id}")
async def get_ocr_streetview_url(pano_id: str, ocr_id: int):
    """Generate a Google Street View URL for an OCR result, taking into account OCR coordinates."""
    conn = None
    try:
        logger.info(f"Looking up OCR result {ocr_id} for panorama ID: {pano_id}")
        conn = get_db()
        cursor = conn.cursor()

        # Get OCR result and panorama data
        cursor.execute(
            """
            SELECT 
                ocr.id,
                ocr.pano_id,
                ocr.text,
                ocr.confidence,
                ocr.yaw,
                ocr.pitch,
                ocr.width,
                ocr.height,
                ocr.engine,
                sp.lat,
                sp.lon,
                sp.heading,
                sp.pitch as panorama_pitch,
                sp.roll
            FROM ocr_result ocr
            JOIN search_panoramas sp ON ocr.pano_id = sp.pano_id
            WHERE ocr.pano_id = ? AND ocr.id = ?
        """,
            (pano_id, ocr_id),
        )

        result = cursor.fetchone()
        if not result:
            logger.error(f"OCR result {ocr_id} not found for panorama {pano_id}")
            raise HTTPException(
                status_code=404,
                detail=f"OCR result {ocr_id} not found for panorama {pano_id}",
            )

        # Convert to dict for easier access
        data = dict(result)

        # Generate Street View URL using the OCR coordinates
        gsv_props = get_google_streetview_props(
            panorama_id=data["pano_id"],
            lat=data["lat"],
            lng=data["lon"],
            ocr_yaw=data["yaw"],
            ocr_pitch=data["pitch"],
            street_view_heading=data["heading"],
            street_view_pitch=data["panorama_pitch"],
            street_view_roll=data["roll"],
            ocr_width=data["width"],
            ocr_height=data["height"],
        )

        url = get_google_streetview_embed_url(gsv_props, GOOGLE_MAP_API_KEY)
        print(url)
        logger.info(f"Generated Street View URL for OCR result {ocr_id}")
        return {"url": url}

    except Exception as e:
        logger.error(f"Error generating Street View URL for OCR result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
