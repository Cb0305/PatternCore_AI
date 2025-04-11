import os
import cv2
import numpy as np
import easyocr
import logging
from typing import List, Dict, Tuple

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Standard roulette color mappings
RED_NUMBERS = {
    1, 3, 5, 7, 9, 12, 14, 16, 18,
    19, 21, 23, 25, 27, 30, 32, 34, 36
}
BLACK_NUMBERS = set(range(1, 37)) - RED_NUMBERS


class OCRProcessor:
    def __init__(self, lang: str = 'en', gpu: bool = True, confidence_threshold: float = 0.5):
        """
        Initialize EasyOCR reader for number extraction.
        """
        self.reader = easyocr.Reader([lang], gpu=gpu)
        self.confidence_threshold = confidence_threshold

    def _determine_color(self, number: int) -> str:
        """
        Map number to roulette color.
        """
        if number == 0:
            return "green"
        elif number in RED_NUMBERS:
            return "red"
        elif number in BLACK_NUMBERS:
            return "black"
        return "unknown"

    def extract_from_image(self, image_path: str) -> List[Tuple[int, str]]:
        """
        Extract numbers and corresponding colors from a single image.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image not loaded. Check file path.")
            results = self.reader.readtext(image)

            outcomes = []
            for (bbox, text, conf) in results:
                text = text.strip()
                if conf >= self.confidence_threshold and text.isdigit():
                    num = int(text)
                    if 0 <= num <= 36:
                        color = self._determine_color(num)
                        outcomes.append((num, color))

            logging.info(f"Extracted {len(outcomes)} outcomes from {os.path.basename(image_path)}")
            return outcomes

        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return []

    def batch_process(self, input_dir: str) -> Dict[str, List[Tuple[int, str]]]:
        """
        Process all images in a directory and return filename-to-outcome mapping.
        """
        all_outcomes = {}
        supported_exts = ('.png', '.jpg', '.jpeg')

        for filename in sorted(os.listdir(input_dir)):
            if filename.lower().endswith(supported_exts):
                full_path = os.path.join(input_dir, filename)
                outcomes = self.extract_from_image(full_path)
                all_outcomes[filename] = outcomes

        logging.info(f"Finished OCR on {len(all_outcomes)} files.")
        return all_outcomes
