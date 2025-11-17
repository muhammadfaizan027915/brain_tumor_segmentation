from prisma import Prisma

db = Prisma()
_is_connected = False

async def connect():
    global _is_connected
    if not _is_connected:
        await db.connect()
        _is_connected = True

async def disconnect():
    global _is_connected
    if _is_connected:
        await db.disconnect()
        _is_connected = False
