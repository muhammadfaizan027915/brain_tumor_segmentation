import os
import sys
import shutil
import asyncio


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'web'))


from core.repositories.session_repo import SessionRepo
from core.repositories.mri_upload_repo import MriUploadRepo
from core.repositories.mri_result_repo import MriResultRepo
from utils.prisma_client import connect, disconnect
from config import UPLOAD_FOLDER, RESULT_FOLDER


async def main():
    await connect()
    ended_sessions = await SessionRepo.get_ended_sessions()
    print(f"Found {len(ended_sessions)} ended sessions to clean up.")

    for ended_session in ended_sessions:
        session_id = ended_session.id
        print(f"\nCleaning session: {session_id}")

        try:
            upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
            result_dir = os.path.join(RESULT_FOLDER, session_id)

            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir)
                print(f"Deleted upload folder: {upload_dir}")
            else:
                print(f"Upload folder does not exist: {upload_dir}")

            if os.path.exists(result_dir):
                shutil.rmtree(result_dir)
                print(f"Deleted result folder: {result_dir}")
            else:
                print(f"Result folder does not exist: {result_dir}")

            await MriUploadRepo.delete_mri_uploads_by_session_id(session_id)
            print(f"Deleted MRI upload records for session: {session_id}")

            await MriResultRepo.delete_mri_results_by_session_id(session_id)
            print(f"Deleted MRI result records for session: {session_id}")

            await SessionRepo.delete_session(session_id)
            print(f"Deleted session record: {session_id}")

        except Exception as e:
            print(f"⚠️  Failed to clean session {session_id}: {e}")
            
    await disconnect()
    print("\nCleanup completed.")


if __name__ == "__main__":
    asyncio.run(main())