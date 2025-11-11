# -------------------------------------------------
# predict.py – PRODUCTION READY, DB-INTEGRATED, USER-SPECIFIC
# -------------------------------------------------
import os
import json
import torch
import pandas as pd
import numpy as np
from torch.nn.functional import softmax
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from datetime import datetime, timezone
import argparse
import logging
import sys

# -------------------------------------------------
# ARGUMENTS
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Predict drug-related messages from cleaned_messages")
parser.add_argument('--model_path', type=str, default="ml/xlmr-large/best",
                    help="Path to fine-tuned model (use forward slashes)")
parser.add_argument('--user_id', type=str, required=True,
                    help="User ID to analyze")
parser.add_argument('--batch_size', type=int, default=32,
                    help="Batch size for inference")
parser.add_argument('--min_prob', type=float, default=0.7,
                    help="Min probability to flag as drug-related")
parser.add_argument('--scan_id', type=str, default=None,
                    help="Current scan ID (optional, used to link predictions)")
args = parser.parse_args()  # ← Only once

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "social_scraper"
RAW_COLLECTION = "cleaned_messages"
PREDICTION_COLLECTION = "predictions"
SUMMARY_COLLECTION = "scan_summaries"

MODEL_PATH = args.model_path.replace("\\", "/")  # ← Force forward slashes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = args.batch_size
MIN_PROB = args.min_prob
USER_ID = args.user_id

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),  # ← Log to stderr
        logging.FileHandler("logs/predictor.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("predictor")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}...")
try:
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

# -------------------------------------------------
# MONGODB SETUP
# -------------------------------------------------
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    raw_col = db[RAW_COLLECTION]
    pred_col = db[PREDICTION_COLLECTION]
    summary_col = db[SUMMARY_COLLECTION]
    scans_col = db["scans"]  # ← Required for scan_id fallback
    logger.info("Connected to MongoDB.")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    sys.exit(1)

# -------------------------------------------------
# FETCH CLEANED MESSAGES FOR USER
# -------------------------------------------------
def fetch_messages():
    query = {"user_id": USER_ID}
    if args.scan_id:
        query["scan_id"] = args.scan_id
    projection = {
        "_id": 1,
        "cleaned_message": 1,
        "platform": 1,
        "sender": 1,
        "timestamp": 1
    }
    cursor = raw_col.find(query, projection)
    messages = list(cursor)
    logger.info(f"Fetched {len(messages)} cleaned messages for user_id: {USER_ID}")
    if messages:
        sample = messages[0].get("cleaned_message", "")[:100]
        logger.info(f"Sample message: '{sample}'")
    return messages

# -------------------------------------------------
# BATCHED PREDICTION
# -------------------------------------------------
def predict_batch(texts):
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**encodings)
        probs = softmax(outputs.logits, dim=1).cpu().numpy()

    preds = (probs[:, 1] >= MIN_PROB).astype(int)
    return preds, probs[:, 1]

# -------------------------------------------------
# MAIN PREDICTION LOOP
# -------------------------------------------------
def run_prediction():
    messages = fetch_messages()
    start_time = datetime.now(timezone.utc)

    if not messages:
        logger.info("No messages to predict – writing zero-summary.")
        summary = {
            "user_id": USER_ID,
            "total_messages": 0,
            "threats_detected": 0,
            "suspicious_activities": 0,
            "clean_messages": 0,
            "scan_time_seconds": 0.0,
            "model": os.path.basename(os.path.dirname(MODEL_PATH)),
            "threshold": MIN_PROB,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "live_scan"
        }
        summary_col.replace_one({"user_id": USER_ID}, summary, upsert=True)
        return summary

    # -------------------------------------------------
    # DETERMINE SCAN_ID (from CLI or latest scan)
    # -------------------------------------------------
    SCAN_ID = args.scan_id
    if not SCAN_ID:
        latest_scan = scans_col.find_one(
            {"user_id": USER_ID},
            sort=[("timestamp", -1)]
        )
        SCAN_ID = str(latest_scan["_id"]) if latest_scan else None
        logger.info(f"No scan_id provided. Using latest scan: {SCAN_ID}")
    else:
        logger.info(f"Using provided scan_id: {SCAN_ID}")

    # -------------------------------------------------
    # PREPARE DATA
    # -------------------------------------------------
    texts = [m["cleaned_message"] for m in messages]
    ids = [str(m["_id"]) for m in messages]
    platforms = [m.get("platform", "unknown") for m in messages]
    senders = [m.get("sender", "unknown") for m in messages]
    timestamps = [m.get("timestamp", datetime.now(timezone.utc).isoformat()) for m in messages]

    predictions = []
    probabilities = []
    logger.info(f"Starting batched prediction on {len(texts)} messages...")

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_preds, batch_probs = predict_batch(batch_texts)
        predictions.extend(batch_preds)
        probabilities.extend(batch_probs)

    scan_time = (datetime.now(timezone.utc) - start_time).total_seconds()

    # -------------------------------------------------
    # SAVE PER-MESSAGE PREDICTIONS
    # -------------------------------------------------
    bulk_ops = []
    threats = suspicious = clean = 0

    for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
        doc_id = ids[idx]
        is_threat = pred == 1
        is_suspicious = 0.3 <= prob < MIN_PROB

        if is_threat:
            threats += 1
        elif is_suspicious:
            suspicious += 1
        else:
            clean += 1

        pred_doc = {
            "message_id": doc_id,
            "user_id": USER_ID,
            "scan_id": SCAN_ID,
            "platform": platforms[idx],
            "sender": senders[idx],
            "sender_id": str(messages[idx].get("_id")),  # Unique per message
            "original_message": texts[idx],
            "timestamp": timestamps[idx],
            "probability": round(float(prob), 4),
            "label": ("drug-related" if is_threat else
                      "suspicious" if is_suspicious else "clean"),
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "model_used": MODEL_PATH
        }

        bulk_ops.append(
            UpdateOne(
                {"message_id": doc_id, "user_id": USER_ID},
                {"$set": pred_doc},
                upsert=True
            )
        )

    if bulk_ops:
        try:
            result = pred_col.bulk_write(bulk_ops, ordered=False)
            logger.info(f"Saved {result.upserted_count + result.modified_count} predictions.")
        except Exception as e:
            logger.error(f"Bulk write failed: {e}")

    # -------------------------------------------------
    # BUILD & PERSIST SUMMARY
    # -------------------------------------------------
    platform = messages[0].get("platform", "unknown") if messages else "unknown"
    summary = {
        "user_id": USER_ID,
        "scan_id": SCAN_ID,
        "platform": platform,
        "total_messages": len(messages),
        "threats_detected": threats,
        "suspicious_activities": suspicious,
        "clean_messages": clean,
        "scan_time_seconds": round(scan_time, 2),
        "model": os.path.basename(os.path.dirname(MODEL_PATH)),
        "threshold": MIN_PROB,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "live_scan"
    }

    try:
        summary_col.replace_one({"user_id": USER_ID, "scan_id": SCAN_ID}, summary, upsert=True)
        logger.info("Summary written to scan_summaries.")
    except Exception as e:
        logger.error(f"Failed to write summary: {e}")

    logger.info(f"Prediction complete – {threats} threats, {suspicious} suspicious, {clean} clean.")
    return summary

# -------------------------------------------------
# CLI ENTRYPOINT
# -------------------------------------------------
if __name__ == "__main__":
    result = run_prediction()

    # --- HUMAN READABLE ---
    print("\n" + "="*60, file=sys.stderr)
    print("PREDICTION SUMMARY", file=sys.stderr)
    print("="*60, file=sys.stderr)
    for k, v in result.items():
        print(f"{k.replace('_', ' ').title():25}: {v}", file=sys.stderr)
    print("="*60, file=sys.stderr)

    # --- CRITICAL: JSON OUTPUT FOR FLASK (on stdout) ---
    print("\n--- JSON OUTPUT FOR API ---", flush=True)
    json_output = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_output, flush=True)

    # Force flush
    sys.stdout.flush()
    sys.stderr.flush()