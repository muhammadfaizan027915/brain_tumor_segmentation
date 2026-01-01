from utils import db
from core.models import MRIFile


class MriUploadRepo:
    @staticmethod
    async def create_mri_upload(session_id: str, upload_dir_id: str, mri_file: MRIFile):
        return await db.mriupload.create(
            data={
                "filename": mri_file.name,
                "filepath": mri_file.filepath,
                "user_session_id": session_id,
                "upload_dir_id": upload_dir_id
            }
        )

    @staticmethod
    async def delete_mri_upload(upload_id: int):
        return await db.mriupload.delete(where={"id": upload_id})

    @staticmethod
    async def delete_mri_uploads_by_session_id(session_id: str):
        return await db.mriupload.delete_many(where={"user_session_id": session_id})
