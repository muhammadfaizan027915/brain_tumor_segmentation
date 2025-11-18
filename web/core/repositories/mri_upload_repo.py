from utils.prisma_client import db, connect
from core.models.mri_file import MRIFile

class MriUploadRepo:
    @staticmethod
    async def create_mri_upload(session_id: str, mri_file: MRIFile):
        await connect()
        return await db.mriupload.create(
            data={
                "filename": mri_file.name,
                "filepath": mri_file.filepath,
                "user_session_id": session_id
            }
        )

    @staticmethod
    async def delete_mri_upload(upload_id: int):
        await connect()
        return await db.mriupload.delete(where={"id": upload_id})
