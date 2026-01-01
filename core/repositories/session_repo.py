from utils.prisma_client import db
from datetime import datetime


class SessionRepo:
    @staticmethod
    async def get_session(session_id: str):
        return await db.usersession.find_first(
            where={
                "id": session_id,
                "endedat": None
            }
        )

    @staticmethod
    async def initialize_session():
        return await db.usersession.create(data={})

    @staticmethod
    async def end_session(session_id: str):
        return await db.usersession.update(
            where={"id": session_id},
            data={"endedat": datetime.now()}
        )

    async def get_ended_sessions():
        return await db.usersession.find_many(
            where={
                "endedat": {
                    "not": None
                }
            }
        )

    @staticmethod
    async def delete_session(session_id: str):
        return await db.usersession.delete(where={"id": session_id})
