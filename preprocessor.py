import os
import re
import csv
import logging
import logging.handlers  # ✅ Needed for TimedRotatingFileHandler
from tqdm import tqdm
from pymongo import MongoClient, UpdateOne
import spacy
from datetime import datetime, timedelta, timezone  # ✅ timezone-aware UTC
import argparse

# -------------------------------------
# ARGUMENT PARSER
# -------------------------------------
parser = argparse.ArgumentParser(description="Preprocess scraped data")
parser.add_argument('--user_id', required=True, help="User ID from Flask")
parser.add_argument('--export_csv', action='store_true', help="Export to CSV after processing")
args = parser.parse_args()

# -------------------------------------
# CONFIG
# -------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "social_scraper"
RAW_COLLECTION = "raw_messages"
OUTPUT_COLLECTION = "cleaned_messages"
CSV_BACKUP = "cleaned_messages.csv"
BULK_BATCH_SIZE = 500

# -------------------------------------
# LOGGING CONFIG
# -------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.TimedRotatingFileHandler(
            'logs/preprocessor.log', when='midnight', backupCount=14, encoding='utf-8'
        )
    ]
)

# -------------------------------------
# MONGODB CONNECTION
# -------------------------------------
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    output_col = db[OUTPUT_COLLECTION]
    logging.info("✅ Connected to MongoDB successfully.")
except Exception as e:
    logging.exception(f"❌ MongoDB connection failed: {e}")
    raise SystemExit(1)

# -------------------------------------
# LOAD SPACY MODEL (Multilingual)
# -------------------------------------
try:
    nlp = spacy.load("xx_sent_ud_sm", disable=["ner", "parser"])
except Exception:
    logging.warning("⚠️ Spacy model not found, using blank 'en' model.")
    nlp = spacy.blank("en")

# -------------------------------------
# REGEX PATTERNS
# -------------------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile("[" +
    u"\U0001F600-\U0001F64F" +
    u"\U0001F300-\U0001F5FF" +
    u"\U0001F680-\U0001F6FF" +
    u"\U0001F1E0-\U0001F1FF" +
    "]+", flags=re.UNICODE)
CURRENCY_MAP = {"₹": " rs ", "$": " usd ", "€": " eur ", "£": " gbp "}

# -------------------------------------
# TEXT CLEANING FUNCTIONS
# -------------------------------------
def clean_text(text: str) -> str:
    """Remove noise, URLs, mentions, emojis, normalize to lowercase and ascii-ish."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    if not text:
        return ""

    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = EMOJI_RE.sub(" ", text)

    for sym, repl in CURRENCY_MAP.items():
        text = text.replace(sym, repl)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text: str) -> str:
    """Lemmatize using multilingual spaCy. Returns lowercased string."""
    if not text:
        return ""
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower() if token.lemma_ else token.text.lower()
        for token in doc if not token.is_punct and not token.is_space
    ])

# -------------------------------------
# MAIN PROCESSOR
# -------------------------------------
def process_collection(user_id=None):
    """
    Process messages in the raw collection, clean them, and store in output collection.
    Returns number of processed messages.
    user_id: Optional, to process only user-specific data.
    """
    collection_name = RAW_COLLECTION
    logging.info(f"🔍 Processing collection: {collection_name}")
    collection = db[collection_name]
    query = {"user_id": user_id} if user_id else {}
    cursor = collection.find(query, {"_id": 0, "message": 1, "platform": 1, "timestamp": 1, "sender": 1})

    # Use .count_documents() to check emptiness
    if collection.count_documents(query) == 0:
        logging.info(f"[Preprocessor] No messages found in {collection_name}")
        return 0

    batch_ops = []
    processed_count = 0

    for doc in tqdm(cursor, desc=f"{collection_name}"):
        raw_text = str(doc.get("message", "")).strip()
        if not raw_text:
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        lemmatized = lemmatize_text(cleaned)
        if not lemmatized:
            continue

        platform = (doc.get("platform") or "unknown").title()
        timestamp = doc.get("timestamp") or datetime.now(timezone.utc)
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        else:
            try:
                timestamp = datetime.fromisoformat(timestamp).isoformat()
            except ValueError:
                timestamp = str(timestamp)
        sender = doc.get("sender") or "Unknown"

        cleaned_doc = {
            "source_collection": collection_name,
            "platform": platform,
            "sender": sender,
            "timestamp": timestamp,
            "cleaned_message": lemmatized,
            "processed_at": datetime.now(timezone.utc)
        }
        if user_id:
            cleaned_doc["user_id"] = user_id

        batch_ops.append(UpdateOne(
            {
                "source_collection": collection_name,
                "timestamp": timestamp,
                "sender": sender,
                "cleaned_message": lemmatized
            },
            {"$set": cleaned_doc},
            upsert=True
        ))

        processed_count += 1
        if len(batch_ops) >= BULK_BATCH_SIZE:
            try:
                output_col.bulk_write(batch_ops, ordered=False)
            except Exception as e:
                logging.exception(f"Bulk write failed for {collection_name}: {e}")
            batch_ops.clear()

    if batch_ops:
        try:
            output_col.bulk_write(batch_ops, ordered=False)
        except Exception as e:
            logging.exception(f"Bulk write failed for {collection_name}: {e}")

    logging.info(f"✅ {collection_name}: {processed_count} cleaned messages stored.")
    return processed_count

# -------------------------------------
# CSV BACKUP (Clean Only)
# -------------------------------------
def export_to_csv(user_id=None):
    """Export cleaned messages to CSV, optionally filtered by user_id."""
    query = {"user_id": user_id} if user_id else {}
    cursor = output_col.find(query, {"_id": 0, "source_collection": 1, "platform": 1, "timestamp": 1, "sender": 1, "cleaned_message": 1, "processed_at": 1})
    fieldnames = ["source_collection", "platform", "timestamp", "sender", "cleaned_message", "processed_at"]
    try:
        with open(CSV_BACKUP, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in cursor:
                pa = row.get("processed_at")
                row["processed_at"] = pa.isoformat() if isinstance(pa, datetime) else str(pa) if pa else ""
                ts = row.get("timestamp")
                row["timestamp"] = ts.isoformat() if isinstance(ts, datetime) else str(ts) if ts else ""
                safe_row = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(safe_row)
        logging.info(f"📁 Exported cleaned messages to {CSV_BACKUP}")
    except Exception as e:
        logging.exception(f"Failed to export to CSV: {e}")

# -------------------------------------
# ENTRY POINT
# -------------------------------------
if __name__ == "__main__":
    logging.info("🚀 Starting preprocessing pipeline (CLEAN ONLY MODE)...")
    try:
        output_col.drop()
        logging.info(f"🧹 Dropped old collection: '{OUTPUT_COLLECTION}'")
    except Exception as e:
        logging.exception(f"Failed to drop old collection: {e}")

    process_collection(args.user_id)
    if args.export_csv:
        export_to_csv(args.user_id)
    logging.info("🏁 Preprocessing completed — only cleaned messages stored in MongoDB.")
