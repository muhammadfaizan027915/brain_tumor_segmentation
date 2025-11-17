from utils.prisma_client import db, connect
from datetime import datetime

class SessionRepo:
    @staticmethod
    async def initialize_session():
        await connect()
        return await db.usersession.create(data={})
    
    @staticmethod
    async def end_session(session_id: str):
        await connect()
        return await db.usersession.update(
            where={"id": session_id},
            data={"endedat": datetime.now()}
        )
