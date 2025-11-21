from utils.prisma_client import db, connect
from core.models.segmentation_result import SegmentationResult


class MriResultRepo:
    @staticmethod
    async def create_mri_result(session_id: str, upload_dir_id: str, result: SegmentationResult):
        await connect()
        return await db.mriresult.create(
            data={
                "output_path": result.filepath,
                "user_session_id": session_id,
                "upload_dir_id": upload_dir_id
            }
        )

    @staticmethod
    async def delete_mri_result(upload_id: int):
        await connect()
        return await db.mriresult.delete(where={"id": upload_id})

    @staticmethod
    async def delete_mri_results_by_session_id(session_id: str):
        await connect()
        return await db.mriresult.delete_many(where={"user_session_id": session_id})
