from telethon import TelegramClient
import asyncio, os

api_id = 25178035
api_hash = "04892392fbd98c931c1091309a96b026"
phone = "+919372867657"

async def main():
    client = TelegramClient("anon", api_id, api_hash)
    await client.start(phone=phone)
    print("Session saved → anon.session")
    await client.disconnect()

asyncio.run(main())