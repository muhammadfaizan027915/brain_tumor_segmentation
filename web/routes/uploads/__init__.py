from quart import Blueprint, request, redirect, url_for, render_template
from core.repositories.session_repo import SessionRepo
from core.repositories.mri_upload_repo import MriUploadRepo
from core.handlers.mri_file_upload_handler import MRIFileUploadHandler

uploads_bp = Blueprint("Uploads", __name__)

@uploads_bp.route("/", methods=["GET"])
async def render_uploads_page():
    return await render_template("uploads.html", active="uploads") 

@uploads_bp.route("/upload_file", methods=["POST"])
async def upload_file():
    file = await request.files
    file_part = file.get("file")

    if not file_part:
        return redirect(url_for("Uploads.render_uploads_page"))

    file_handler = MRIFileUploadHandler()
    
    # 1️⃣ Initialize session
    session = await SessionRepo.initialize_session()
    session_id = session.id

    # 2️⃣ Validate & store file
    mri_file, errors = file_handler.validate_and_upload(file_part, session_id)
    
    if errors:
        print("Validation Errors:", errors)
        return redirect(url_for("Uploads.render_uploads_page"))

    # 3️⃣ Save upload record
    await MriUploadRepo.create_mri_upload(session_id, mri_file)

    # 4️⃣ Redirect to results
    return redirect(url_for("Results.render_results_page", session_id=session_id)) 