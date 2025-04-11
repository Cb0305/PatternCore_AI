import os
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import sys

# Include OCRProcessor path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ocr.ocr_processor import OCRProcessor

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Config ===
INPUT_DIR = "../../data/screenshots"
OUTPUT_FILE = "../../data/processed_csvs/target_outcomes_master.csv"

# Define image groups per batch
BATCH_GROUPS = {
    "batch_1": ["ss101.png", "ss102.png", "ss103.png", "ss104.png"],
    "batch_2": ["ss105.png", "ss106.png", "ss107.png"],
    "batch_3": ["ss108.png", "ss109.png", "ss110.png"],
    "batch_4": ["ss111.png", "ss112.png", "ss113.png"],
    "batch_5": ["ss114.png", "ss115.png", "ss116.png"],
    "batch_6": ["ss117.png", "ss118.png", "ss119.png"],
    "batch_7": ["ss120.png", "ss121.png", "ss122.png"],
    "batch_8": ["ss123.png", "ss124.png", "ss125.png"],
}

# === Helpers ===
def generate_fake_timestamps(count: int, start_time=None) -> List[str]:
    """
    Generate fake timestamps spaced every 3 seconds.
    """
    if start_time is None:
        start_time = datetime.now()
    return [(start_time + timedelta(seconds=i * 3)).strftime("%Y-%m-%d %H:%M:%S") for i in range(count)]


def merge_ocr_data(ocr_output: Dict[str, List[Tuple[int, str]]]) -> List[Dict]:
    """
    Merge OCR results with fake timestamps and batch info.
    """
    merged_rows = []
    for batch_name, image_files in BATCH_GROUPS.items():
        batch_outcomes = []
        for fname in image_files:
            if fname in ocr_output:
                batch_outcomes.extend(ocr_output[fname])
            else:
                logging.warning(f"{fname} not found in OCR output.")

        timestamps = generate_fake_timestamps(len(batch_outcomes))
        for i, (number, color) in enumerate(batch_outcomes):
            merged_rows.append({
                "timestamp": timestamps[i],
                "outcome": number,
                "color": color,
                "batch": batch_name
            })

        logging.info(f"{batch_name}: Merged {len(batch_outcomes)} outcomes.")
    return merged_rows


def save_to_csv(rows: List[Dict], output_path: str):
    """
    Save merged rows to the CSV output file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "outcome", "color", "batch"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Saved {len(rows)} outcomes to {output_path}")


def run_cleaner():
    """
    Full OCR + merge + save pipeline.
    """
    ocr = OCRProcessor()
    ocr_results = ocr.batch_process(INPUT_DIR)
    merged_rows = merge_ocr_data(ocr_results)
    save_to_csv(merged_rows, OUTPUT_FILE)


# Entry point
if __name__ == "__main__":
    run_cleaner()
