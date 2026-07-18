import os
import shutil
import tempfile
import threading
import traceback
import zipfile

from flask import Flask, request, render_template, send_file, flash, redirect, url_for

import pipeline

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# Reference data (periodic table, bond-distance stats) already ships in the
# repo alongside app.py — no need to fetch it over the network at all.
# setup_reference_data() only downloads a file if it's missing locally, so
# pointing it at this directory just loads what's already deployed.
REFERENCE_CACHE = os.path.dirname(os.path.abspath(__file__))
pipeline.setup_reference_data(REFERENCE_CACHE)

# run_pipeline() uses os.chdir() internally, which is process-wide state.
# A lock keeps concurrent requests from stepping on each other's working
# directory. Fine for a low-traffic tool; each run also isn't fast.
_PIPELINE_LOCK = threading.Lock()

# Where finished result .zip files are kept so /download/<id> can serve them.
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", elements=pipeline.supported_elements())


@app.route("/run", methods=["POST"])
def run():
    cif_file = request.files.get("cif_file")
    if not cif_file or cif_file.filename == "":
        flash("Please upload a CIF file.")
        return redirect(url_for("index"))

    element = request.form.get("element", "Na")
    max_path_length = float(request.form.get("max_path_length", 7))
    nimages = int(request.form.get("nimages", 5))
    create_folder = request.form.get("create_folder", "No")

    oxi = pipeline.oxidation_state_for(element)
    if oxi is None:
        flash(f"No oxidation state configured for element '{element}'.")
        return redirect(url_for("index"))

    # ICSD/structure name derived from the uploaded filename, e.g.
    # "29962.cif" -> "29962", or "29962_ord_rama.cif" -> "29962_ord_rama"
    original_name = cif_file.filename
    st_name = os.path.splitext(original_name)[0]

    work_dir = tempfile.mkdtemp(prefix="dc_")
    cif_path = os.path.join(work_dir, f"{st_name}.cif")
    cif_file.save(cif_path)

    original_cwd = os.getcwd()
    try:
        with _PIPELINE_LOCK:
            os.chdir(work_dir)
            try:
                df_out, low_energy_df, dim_struct, dim_path, dim_stddev, _, structure = (
                    pipeline.run_pipeline(
                        element_symbol=element,
                        element_oxi=oxi,
                        st_name=st_name,
                        max_path_length=max_path_length,
                        nimages=nimages,
                        create_folder=create_folder,
                    )
                )
            finally:
                os.chdir(original_cwd)
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        flash(f"Pipeline failed: {e}")
        traceback.print_exc()
        return redirect(url_for("index"))

    # Build the 3x6 descriptor summary (same as the notebook's Section 5)
    def _get_dim(d, dim):
        val = d.get(dim, None)
        if val is None:
            return None
        try:
            return round(float(val), 2)
        except (TypeError, ValueError):
            return val

    summary_rows = []
    for label, dim_dict in [("struct", dim_struct), ("path", dim_path), ("stddev", dim_stddev)]:
        summary_rows.append({
            "reference": label,
            "0-D": _get_dim(dim_dict, 0),
            "1-D": _get_dim(dim_dict, 1),
            "2-D": _get_dim(dim_dict, 2),
            "3-D": _get_dim(dim_dict, 3),
        })

    # Zip every output file this run produced so the user can download it all.
    zip_name = f"{st_name}_results.zip"
    zip_path = os.path.join(RESULTS_DIR, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(work_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, work_dir)
                zf.write(fpath, arcname)

    shutil.rmtree(work_dir, ignore_errors=True)

    return render_template(
        "results.html",
        st_name=st_name,
        composition=str(structure.composition),
        summary_rows=summary_rows,
        table=low_energy_df.round(2).to_dict(orient="records"),
        table_columns=[c for c in
            ["NEB", "NEB1", "Category_stddev", "Max_contraction_stddev",
             "Category_path", "Max_contraction_path", "Perc"]
            if c in low_energy_df.columns],
        zip_name=zip_name,
    )


@app.route("/download/<zip_name>")
def download(zip_name):
    # basic guard: only serve files we generated into RESULTS_DIR
    safe_name = os.path.basename(zip_name)
    path = os.path.join(RESULTS_DIR, safe_name)
    if not os.path.exists(path):
        flash("That results file is no longer available — please re-run.")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name=safe_name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
