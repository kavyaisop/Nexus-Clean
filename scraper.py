# scraper.py – FINAL LIVE-STREAMING VERSION (DEC 2025)
# -------------------------------------------------
# PRINTS JSON TO STDOUT → app.py STREAMS LIVE
# -------------------------------------------------
import os
import re
import json
import time
import asyncio
import logging
import argparse
import random
import hmac
import hashlib
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from telethon import TelegramClient
import requests
import csv

try:
    from docx import Document
except ImportError:
    Document = None

from pdfminer.high_level import extract_text as extract_pdf_text
import logging.handlers

# -------------------------------------------------
# PROXY SUPPORT
# -------------------------------------------------
PROXY = os.getenv("INSTAGRAM_PROXY")

def _proxies():
    if PROXY:
        return {"http": PROXY, "https": PROXY}
    return None

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "social_scraper")
INSTAGRAM_COOKIE_FILE = "insta_cookies.json"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID", "25178035"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "04892392fbd98c931c1091309a96b026")
TELEGRAM_PHONE = os.getenv("TELEGRAM_PHONE", "+919372867657")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------------------------------
# LOGGING (FILE ONLY)
# -------------------------------------------------
logger = logging.getLogger("scraper")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

file_handler = logging.handlers.TimedRotatingFileHandler(
    os.path.join(LOG_DIR, "scraper.log"), when="midnight", backupCount=14, encoding="utf-8"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -------------------------------------------------
# LIVE PROGRESS OUTPUT (STDOUT)
# -------------------------------------------------
def log_progress(msg_type: str, data: dict):
    """Print structured JSON for app.py to stream live"""
    print(json.dumps({"type": msg_type, "data": data}, default=str, ensure_ascii=False), flush=True)

# -------------------------------------------------
# MONGODB
# -------------------------------------------------
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    logger.info("MongoDB connected")
except ConnectionFailure as e:
    logger.error(f"MongoDB unreachable: {e}")
    raise SystemExit(1)
except Exception as e:
    logger.error(f"MongoDB connection error: {e}")
    raise SystemExit(1)

db = mongo_client[DB_NAME]
raw_messages_col = db["raw_messages"]

def insert_one(doc: dict) -> bool:
    try:
        raw_messages_col.insert_one(doc)
        return True
    except Exception as e:
        logger.error(f"MongoDB insert error: {e}")
        return False

# -------------------------------------------------
# CHECKPOINTS
# -------------------------------------------------
def _cp_path(name: str, scan_id: Optional[str] = None) -> str:
    base = name
    if scan_id:
        base += f"_{scan_id}"
    return os.path.join(CHECKPOINT_DIR, f"{base}.json")

def load_checkpoint(name: str, scan_id: Optional[str] = None) -> dict:
    p = _cp_path(name, scan_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {p}: {e}")
    return {}

def save_checkpoint(name: str, data: dict, scan_id: Optional[str] = None):
    p = _cp_path(name, scan_id)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Failed to save checkpoint {p}: {e}")

# -------------------------------------------------
# INSTAGRAM: X-IG-WWW-Claim
# -------------------------------------------------
def generate_ig_www_claim(sessionid: str, ig_did: str) -> str:
    key = b"5723a378d4e2f94e6f3c7a1d8b9e0c1d"
    message = f"{sessionid}{ig_did}"
    digest = hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"hmac.{digest}"

# -------------------------------------------------
# INSTAGRAM SESSION
# -------------------------------------------------
def make_instagram_session() -> Optional[requests.Session]:
    if not os.path.exists(INSTAGRAM_COOKIE_FILE):
        logger.error(f"Cookie file missing: {INSTAGRAM_COOKIE_FILE}")
        return None

    try:
        with open(INSTAGRAM_COOKIE_FILE, "r", encoding="utf-8") as f:
            cookies = json.load(f)

        required = ["csrftoken", "sessionid", "ds_user_id"]
        missing = [c for c in required if not any(x["name"] == c for x in cookies)]
        if missing:
            logger.error(f"Missing required cookies: {missing}")
            return None

        s = requests.Session()
        for c in cookies:
            domain = c.get("domain", ".instagram.com")
            path = c.get("path", "/")
            s.cookies.set(c["name"], c["value"], domain=domain, path=path)

        csrftoken = s.cookies.get("csrftoken")
        sessionid = s.cookies.get("sessionid")
        ig_did = s.cookies.get("ig_did", "")
        mid = s.cookies.get("mid", "")

        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "X-CSRFToken": csrftoken,
            "X-IG-App-ID": "936619743392459",
            "X-ASBD-ID": "129477",
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        })

        if ig_did and sessionid:
            s.headers["X-IG-WWW-Claim"] = generate_ig_www_claim(sessionid, ig_did)
        if mid:
            s.headers["X-Mid"] = mid

        original_request = s.request
        def wrapped_request(method, url, **kwargs):
            kwargs.setdefault("proxies", _proxies())
            return original_request(method, url, **kwargs)
        s.request = wrapped_request

        return s
    except Exception as e:
        logger.exception(f"Failed to create Instagram session: {e}")
        return None

# -------------------------------------------------
# LIST: Instagram DMs
# -------------------------------------------------
def list_instagram_chats(user_id: Optional[str] = None) -> List[dict]:
    session = make_instagram_session()
    if not session:
        return []

    chats = []
    cursor = None
    
    try:
        for page in range(5):
            params = {"persistentBadging": "true", "limit": "20"}
            if cursor:
                params["cursor"] = cursor

            r = session.get("https://i.instagram.com/api/v1/direct_v2/inbox/", params=params, timeout=15)
            if r.status_code != 200:
                if page == 0:
                    r = session.get("https://www.instagram.com/api/v1/direct_v2/inbox/", params={"limit": 20}, timeout=15)
                if r.status_code != 200:
                    break

            data = r.json()
            threads = data.get(" threads", [])
            if not threads:
                break

            for thread in threads:
                tid = thread.get("thread_id")
                users = thread.get("users", [])
                title = users[0].get("username", "Unknown") if users else "Unknown"
                last_activity = thread.get("last_activity_at", 0)
                if last_activity > 1_000_000_000_000:
                    last_activity //= 1_000_000
                chats.append({
                    "id": tid,
                    "username": title,
                    "last_activity": datetime.fromtimestamp(last_activity).isoformat() if last_activity else datetime.now().isoformat()
                })

            cursor = data.get("inbox", {}).get("oldest_cursor")
            if not cursor:
                break
            time.sleep(1.5)
    except Exception as e:
        logger.exception(f"List IG DMs error: {e}")

    return chats

# -------------------------------------------------
# LIST: Telegram Groups
# -------------------------------------------------
async def list_telegram_chats(user_id: Optional[str] = None) -> List[dict]:
    client = TelegramClient("anon", TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start(phone=TELEGRAM_PHONE)
    chats = []
    
    try:
        async for dialog in client.iter_dialogs(limit=100):
            if dialog.is_group or dialog.is_channel:
                e = dialog.entity
                chats.append({
                    "id": str(e.id),
                    "title": dialog.name or "Unknown",
                    "members": getattr(e, "participants_count", 0) or 0
                })
                if len(chats) >= 50:
                    break
    except Exception as e:
        logger.exception(f"List Telegram error: {e}")
    finally:
        await client.disconnect()
    
    return chats

# -------------------------------------------------
# SCRAPE: Instagram Hashtag
# -------------------------------------------------
def scrape_instagram_hashtag(hashtag: str, user_id: Optional[str], scan_id: Optional[str]) -> int:
    hashtag = hashtag.lstrip('#').lower()
    key = f"ig_hashtag_{hashtag}"
    cp = load_checkpoint(key, scan_id)
    seen = set(cp.get("seen", []))
    saved = 0
    
    session = make_instagram_session()
    if not session:
        return 0

    try:
        api_url = "https://www.instagram.com/api/v1/tags/web_info/"
        r = session.get(api_url, params={"tag_name": hashtag}, timeout=15)
        if r.status_code != 200:
            log_progress("error", {"error": f"HTTP {r.status_code}"})
            return 0
        
        data = r.json().get("data", [])
        first_tag = data[0] if data else {}
        sections = first_tag.get("sections", [])

        for section in sections:
            medias = section.get("layout_content", {}).get("medias", [])
            for item in medias:
                media = item.get("media", {})
                sc = media.get("code") or media.get("shortcode")
                if not sc or f"{hashtag}_{sc}" in seen:
                    continue

                caption = ""
                cap_obj = media.get("caption")
                if cap_obj:
                    caption = cap_obj.get("text", "") if isinstance(cap_obj, dict) else str(cap_obj)
                if not caption or len(caption) < 3:
                    continue

                seen.add(f"{hashtag}_{sc}")
                ts = datetime.fromtimestamp(media.get("taken_at", time.time()))
                username = media.get("user", {}).get("username", "unknown")

                doc = {
                    "platform": "instagram",
                    "scan_type": "hashtag",
                    "timestamp": ts,
                    "sender": username,
                    "message": caption[:1000],
                    "target": hashtag,
                    "post_url": f"https://www.instagram.com/p/{sc}/",
                    "user_id": user_id,
                    "scan_id": scan_id
                }
                
                if insert_one(doc):
                    saved += 1
                    log_progress("message", {
                        "sender": username,
                        "message": caption[:300],
                        "timestamp": ts.isoformat(),
                        "platform": "instagram"
                    })

        if saved > 0:
            save_checkpoint(key, {"seen": list(seen)}, scan_id)
        return saved
        
    except Exception as e:
        log_progress("error", {"error": str(e)})
        logger.exception(f"Hashtag scrape error: {e}")
        return 0

# -------------------------------------------------
# SCRAPE: Instagram DMs
# -------------------------------------------------
def scrape_instagram_dms(thread_ids: List[str], user_id: Optional[str], scan_id: Optional[str]) -> int:
    session = make_instagram_session()
    if not session:
        return 0

    key = "ig_dms"
    cp = load_checkpoint(key, scan_id)
    seen = set(cp.get("seen", []))
    saved = 0
    
    for tid in thread_ids:
        cursor = None
        for page in range(20):
            params = {"limit": 50}
            if cursor:
                params["cursor"] = cursor

            try:
                r = session.get(f"https://i.instagram.com/api/v1/direct_v2/threads/{tid}/", params=params, timeout=15)
                if r.status_code != 200:
                    break

                data = r.json()
                items = data.get("thread", {}).get("items", [])
                if not items:
                    break

                for item in items:
                    msg_id = item.get("item_id")
                    if not msg_id or msg_id in seen:
                        continue
                    
                    txt = item.get("text", "")
                    if not txt or len(txt) < 3:
                        continue

                    if any(k in txt.lower() for k in ["added", "removed", "changed", "reacted", "call"]):
                        continue

                    seen.add(msg_id)
                    ts_micro = item.get("timestamp", 0)
                    ts = datetime.fromtimestamp(ts_micro // 1_000_000) if ts_micro else datetime.now()

                    doc = {
                        "platform": "instagram",
                        "scan_type": "dms",
                        "timestamp": ts,
                        "sender": str(item.get("user_id", "unknown")),
                        "message": txt,
                        "thread_id": tid,
                        "user_id": user_id,
                        "scan_id": scan_id
                    }
                    
                    if insert_one(doc):
                        saved += 1
                        log_progress("message", {
                            "sender": str(item.get("user_id", "unknown")),
                            "message": txt[:300],
                            "timestamp": ts.isoformat(),
                            "platform": "instagram"
                        })

                cursor = data.get("thread", {}).get("oldest_cursor")
                if not cursor:
                    break
                time.sleep(0.8)
                
            except Exception as e:
                logger.error(f"DM thread {tid} error: {e}")
                break

    save_checkpoint(key, {"seen": list(seen)}, scan_id)
    return saved

# -------------------------------------------------
# SCRAPE: Telegram Groups
# -------------------------------------------------
async def scrape_telegram_groups(group_ids: List[str], user_id: Optional[str], scan_id: Optional[str]) -> int:
    client = TelegramClient("anon", TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start(phone=TELEGRAM_PHONE)
    
    key = "tg_groups"
    cp = load_checkpoint(key, scan_id)
    seen = set(cp.get("seen", []))
    saved = 0

    for gid in group_ids:
        try:
            entity = await client.get_entity(int(gid))
            async for msg in client.iter_messages(entity, limit=1000):
                if msg.id in seen:
                    continue
                if not msg.message or len(msg.message.strip()) < 3:
                    continue

                sender_name = "Unknown"
                try:
                    sender = await msg.get_sender()
                    sender_name = getattr(sender, "username", None) or getattr(sender, "first_name", "Anonymous")
                except:
                    pass

                seen.add(msg.id)
                
                doc = {
                    "platform": "telegram",
                    "scan_type": "group",
                    "timestamp": msg.date,
                    "sender": sender_name,
                    "sender_id": str(msg.sender_id) if msg.sender_id else "",
                    "message": msg.message,
                    "group_id": gid,
                    "user_id": user_id,
                    "scan_id": scan_id
                }
                
                if insert_one(doc):
                    saved += 1
                    log_progress("message", {
                        "sender": sender_name,
                        "message": msg.message[:300],
                        "timestamp": msg.date.isoformat(),
                        "platform": "telegram"
                    })

                await asyncio.sleep(0.1)

            save_checkpoint(key, {"seen": list(seen)}, scan_id)
            
        except Exception as e:
            logger.exception(f"Telegram group {gid} error: {e}")

    await client.disconnect()
    return saved

# -------------------------------------------------
# WHATSAPP FILE PARSER
# -------------------------------------------------
def parse_whatsapp_file(file_path: str, user_id: Optional[str], scan_id: Optional[str]) -> int:
    if not os.path.exists(file_path):
        log_progress("error", {"error": "File not found"})
        return 0

    saved = 0
    text = ""
    
    try:
        ext = file_path.lower().rsplit(".", 1)[-1]
        if ext == "txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == "pdf":
            text = extract_pdf_text(file_path)
        elif ext == "docx" and Document:
            doc = Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            log_progress("error", {"error": f"Unsupported file: {ext}"})
            return 0
            
    except Exception as e:
        log_progress("error", {"error": str(e)})
        return 0

    patterns = [
        r"(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s?[APMapm]{2}) - ([^:]+): (.+)",
        r"\[(\d{1,2}/\d{1,2}/\d{2,4} \d{1,2}:\d{2}:\d{2})\] ([^:]+): (.+)",
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) - ([^:]+): (.+)",
    ]
    
    last_ts = datetime.utcnow()
    lines = text.splitlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        matched = False
        for pat in patterns:
            m = re.match(pat, line)
            if m:
                ts_str, sender, msg = m.groups()
                try:
                    ts = datetime.strptime(ts_str, "%m/%d/%y, %I:%M %p")
                    if ts.year < 2000:
                        ts = ts.replace(year=ts.year + 2000)
                except:
                    try:
                        ts = datetime.strptime(ts_str, "%m/%d/%Y, %I:%M %p")
                    except:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace(" ", "T"))
                        except:
                            ts = last_ts
                last_ts = ts

                doc = {
                    "platform": "whatsapp",
                    "scan_type": "file",
                    "timestamp": ts,
                    "sender": sender.strip(),
                    "message": msg.strip(),
                    "user_id": user_id,
                    "scan_id": scan_id,
                }
                
                if insert_one(doc):
                    saved += 1
                    log_progress("message", {
                        "sender": sender.strip(),
                        "message": msg.strip()[:300],
                        "timestamp": ts.isoformat(),
                        "platform": "whatsapp"
                    })
                matched = True
                break
        
        if not matched and len(line) > 5:
            doc = {
                "platform": "whatsapp",
                "scan_type": "file",
                "timestamp": last_ts,
                "sender": "Unknown",
                "message": line,
                "user_id": user_id,
                "scan_id": scan_id,
            }
            if insert_one(doc):
                saved += 1
                log_progress("message", {
                    "sender": "Unknown",
                    "message": line[:300],
                    "timestamp": last_ts.isoformat(),
                    "platform": "whatsapp"
                })

    return saved

# -------------------------------------------------
# MAIN CLI
# -------------------------------------------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['list_instagram_chats', 'list_telegram_chats', 'scrape'])
    parser.add_argument('--platform', choices=['instagram', 'telegram', 'whatsapp'])
    parser.add_argument('--scan_type', choices=['dms', 'hashtag', 'group', 'file'])
    parser.add_argument('--targets', default='[]')
    parser.add_argument('--target')
    parser.add_argument('--user_id')
    parser.add_argument('--scan_id')
    parser.add_argument('--file_path')

    args = parser.parse_args()

    if args.action == 'list_instagram_chats':
        result = list_instagram_chats(args.user_id)
        print(json.dumps(result, default=str))
        return

    if args.action == 'list_telegram_chats':
        result = await list_telegram_chats(args.user_id)
        print(json.dumps(result, default=str))
        return

    if args.action == 'scrape':
        targets = json.loads(args.targets) if args.targets != '[]' else []
        saved = 0

        try:
            if args.platform == 'instagram':
                if args.scan_type == 'hashtag' and args.target:
                    saved = scrape_instagram_hashtag(args.target, args.user_id, args.scan_id)
                elif args.scan_type == 'dms' and targets:
                    saved = scrape_instagram_dms(targets, args.user_id, args.scan_id)

            elif args.platform == 'telegram' and args.scan_type == 'group' and targets:
                saved = await scrape_telegram_groups(targets, args.user_id, args.scan_id)

            elif args.platform == 'whatsapp' and args.scan_type == 'file' and args.file_path:
                saved = parse_whatsapp_file(args.file_path, args.user_id, args.scan_id)

            print(json.dumps({"saved": saved}))

        except Exception as e:
            log_progress("error", {"error": str(e)})
            print(json.dumps({"saved": 0, "error": str(e)}))

if __name__ == "__main__":
    asyncio.run(main())