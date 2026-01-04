import os
import time
import logging
import requests
import pandas as pd
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv   

# LOAD ENV VARIABLES

load_dotenv()  # Reads the .env file
MAPBOX_TOKEN = os.getenv("MAPBOX_KEY")

if not MAPBOX_TOKEN:
    raise ValueError("MAPBOX_KEY not found in .env file! Please set MAPBOX_KEY=your_token")


# CONFIG
STYLE = "satellite-v9"
ZOOM = 17.5
IMG_SIZE = "512x512"

MAX_WORKERS = 8               # Safe for Mapbox
REQUEST_TIMEOUT = 10
SLEEP_ON_ERROR = 1.0

DATA_FILES = {
    "train": r"C:\Users\KARRA NEHRU\OneDrive\Documents\ml_projects\cdcxyhills_2025\train(1).csv",
    "test":  r"C:\Users\KARRA NEHRU\OneDrive\Documents\ml_projects\cdcxyhills_2025\test(2).csv",
}

OUTPUT_BASE = r"C:\Users\KARRA NEHRU\OneDrive\Documents\ml_projects\cdcxyhills_2025\images"


# LOGGING
logging.basicConfig(
    filename="fetch.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# SESSION WITH RETRIES
def create_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    return session


# IMAGE FETCH
def fetch_image(session, row, split):
    property_id = row["id"]
    lat = row["lat"]
    lon = row["long"]

    output_dir = os.path.join(OUTPUT_BASE, split)
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"{property_id}.jpg")

    # Skip existing images
    if os.path.exists(save_path):
        return "skipped"

    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/{STYLE}/static/"
        f"{lon},{lat},{ZOOM}/{IMG_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return "downloaded"

    except Exception as e:
        logging.error(f"{split} | ID {property_id} | {e}")
        time.sleep(SLEEP_ON_ERROR)
        return "failed"


# MAIN PROCESSING
def process_split(split, csv_path):
    df = pd.read_csv(csv_path)

    session = create_session()
    results = {"downloaded": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(fetch_image, session, row, split)
            for _, row in df.iterrows()
        ]

        for future in tqdm(
            futures,
            desc=f"Fetching {split} images",
            unit="img",
            leave=True
        ):
            status = future.result()
            if status in results:
                results[status] += 1

    print(f"{split.upper()} SUMMARY:", results)

    logging.info(f"{split.upper()} SUMMARY: {results}")
    print(f"{split.upper()} SUMMARY:", results)

def main():
    for split, path in DATA_FILES.items():
        print(f"\n Starting {split} processing...")
        process_split(split, path)


if __name__ == "__main__":
    main()

