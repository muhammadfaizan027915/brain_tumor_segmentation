from prisma import Prisma

db = Prisma()

async def connect():
    if not db.is_connected():
        await db.connect()

async def disconnect():
    if db.is_connected():
        await db.disconnect()
