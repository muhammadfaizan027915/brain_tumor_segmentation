from quart import render_template, Blueprint, request, redirect, url_for
from core.repositories import MriResultRepo
from core.services import SegmentationPredictor
from config import UPLOAD_FOLDER, RESULT_FOLDER

results_bp = Blueprint("Results", __name__)


@results_bp.route("/results", methods=["GET"])
async def render_results_page():
    session_id = request.args.get('session_id')
    upload_dir_id = request.args.get('upload_dir_id')

    if not session_id or not upload_dir_id:
        return redirect(
            url_for("Uploads.render_uploads_page")
        )

    case_path = f"{UPLOAD_FOLDER}/{session_id}/{upload_dir_id}"
    out_path = f"{RESULT_FOLDER}/{session_id}/{upload_dir_id}"
    png_path = f"{out_path}/segmentation.png"

    try:
        predictor = SegmentationPredictor()
        segmentation_result = predictor.predict(
            case_path=case_path, out_path=out_path
        )
        segmentation_result.save(path=png_path)

        result = await MriResultRepo.create_mri_result(
            session_id=session_id,
            upload_dir_id=upload_dir_id,
            result=segmentation_result
        )

        print(f"\nSegmentation PNG saved at: {png_path}")

        return await render_template(
            "results.html", 
            active="results", 
            result=result
        )

    except Exception as e:
        print(f"⚠️ Error during segmentation: {e}")

        return await render_template(
            "results.html",
            active="results",
            errors=[f"Segmentation failed: {str(e)}"]
        )
