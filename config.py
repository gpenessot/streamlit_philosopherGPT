import os
from pathlib import Path
import configparser

# Paths
ROOT = Path(__file__).parent
DATA = ROOT / "data"
MAX_SENTENCE_LENGTH = 100
config = configparser.ConfigParser()
config.read(ROOT / 'config.cfg')

# Qdrant
QDRANT_HOST = config['QDRANT']['host']
QDRANT_PORT = config['QDRANT']['port']
QDRANT_API_KEY = config['QDRANT']['qdrant_api_key']
COLLECTION_NAME='stoician_philosophy'
# OpenAI
OPENAI_API_KEY = config['OPENAI']['openai_api_key']