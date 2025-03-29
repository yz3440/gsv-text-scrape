import os
import time
import sqlite3
import streetview
import concurrent.futures
import os


DB_PATH = "gsv.db"


########################################
# MARK: Database setup
########################################


def setup_database():
    print("Setting up database")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS sample_coords
            (id INTEGER PRIMARY KEY AUTOINCREMENT, lat real, lon real, label text, searched boolean default False)
            """
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS search_panoramas (
        pano_id TEXT PRIMARY KEY,
        lat REAL,
        lon REAL,
        date TEXT,
        copyright TEXT,
        heading REAL,
        pitch REAL,
        roll REAL
    )
    """
    )

    conn.commit()
    conn.close()


########################################
# MARK: Get counts
########################################


def count_unsearched_coords():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM sample_coords WHERE searched = 0")
    res = cursor.fetchone()
    conn.close()
    return res[0]


def count_total_coords():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM sample_coords")
    res = cursor.fetchone()
    conn.close()
    return res[0]


def count_total_panoramas():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM search_panoramas")
    res = cursor.fetchone()
    conn.close()
    return res[0]


def count_panoramas_with_date_and_copyright():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT COUNT(*) FROM search_panoramas WHERE date IS NOT NULL AND date != '' AND copyright IS NOT NULL AND copyright != ''"
    )
    res = cursor.fetchone()
    conn.close()
    return res[0]


if __name__ == "__main__":
    setup_database()

    unsearched_coords = count_unsearched_coords()
    total_coords = count_total_coords()
    searched_coords = total_coords - unsearched_coords
    coord_search_progress = searched_coords / total_coords

    total_panoramas = count_total_panoramas()
    panoramas_with_date_and_copyright = count_panoramas_with_date_and_copyright()
    panoramas_metadata_progress = panoramas_with_date_and_copyright / total_panoramas

    panorama_coords_ratio = total_panoramas / searched_coords
    print("Current Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("\n[Coord Search Progress]")
    print(f"Progress: {coord_search_progress*100:.2f}%")
    print(f"Searched Coords: {searched_coords:,}/{total_coords:,}")

    print("\n[Found Panoramas]")
    print(f"Total Panoramas: {total_panoramas:,}")
    print(f"Panorama to Coord Ratio: {panorama_coords_ratio:.2f} pano/coord")

    print("\n[Panorama Metadata Progress]")
    print(f"Progress: {panoramas_metadata_progress*100:.2f}%")
    print(
        f"Panoramas with Metadata: {panoramas_with_date_and_copyright:,}/{total_panoramas:,}"
    )

    print("\n[Expected Total Panoramas]")
    print(f"Expected Total Panoramas: {total_coords * panorama_coords_ratio:,.0f}")
