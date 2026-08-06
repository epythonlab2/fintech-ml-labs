import json
from pathlib import Path

# Path to the mock data file
DATA_FILE = Path(__file__).parent / "mock_data.json"

# Load the JSON data into a Python dictionary
with open(DATA_FILE, "r", encoding="utf-8") as file:
    ORDERS_DB = json.load(file)