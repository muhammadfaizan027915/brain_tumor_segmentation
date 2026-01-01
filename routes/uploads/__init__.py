from core.repositories import SessionRepo
from core.repositories import MriUploadRepo
from core.handlers import MRIFileUploadHandler

from quart import Blueprint, request, redirect, url_for, render_template, jsonify
from uuid import uuid4

uploads_bp = Blueprint("Uploads", __name__)


@uploads_bp.route("/", methods=["GET"])
async def render_uploads_page():
    session_id = request.args.get("session_id")
    errors_param = request.args.get("errors")
    errors = errors_param.split("|") if errors_param else []

    session = None
    if session_id:
        session = await SessionRepo.get_session(session_id)

    if not session:
        session = await SessionRepo.initialize_session()
        return redirect(
            url_for(
                "Uploads.render_uploads_page",
                session_id=session.id,
                errors=errors
            )
        )

    return await render_template(
        "uploads.html",
        active="uploads",
        session_id=session.id,
        errors=errors
    )


@uploads_bp.route("/end_session", methods=["POST"])
async def end_session():
    data = await request.get_json()
    session_id = data["session_id"]

    await SessionRepo.end_session(session_id=session_id)
    return jsonify({"status": "success"})


@uploads_bp.route("/upload_file", methods=["POST"])
async def upload_file():
    session_id = request.args.get("session_id")

    files_request = await request.files
    files = files_request.getlist("files")

    if not files:
        return redirect(url_for("Uploads.render_uploads_page"))

    upload_dir_id = uuid4().hex
    file_handler = MRIFileUploadHandler()
    uploaded_records = []

    for file in files:
        mri_file, errors = file_handler.validate_and_upload(
            file=file,
            session_id=session_id,
            upload_dir_id=upload_dir_id
        )

        if errors:
            print("Validation Errors:", errors)
            return redirect(
                url_for(
                    "Uploads.render_uploads_page",
                    session_id=session_id,
                    errors="|".join(errors)
                )
            )

        mri_uploaded = await MriUploadRepo.create_mri_upload(session_id, upload_dir_id, mri_file)
        uploaded_records.append(mri_uploaded)

    return redirect(
        url_for(
            "Results.render_results_page",
            session_id=session_id,
            upload_dir_id=upload_dir_id
        )
    )
