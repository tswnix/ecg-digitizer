from __future__ import annotations
import io, json, tempfile, zipfile
from contextlib import redirect_stdout
from pathlib import Path
import cv2
import streamlit as st
from ecg_layout_detection import process_pdf

st.set_page_config(page_title="ECG Scan Digitization", page_icon="📈", layout="wide")
st.title("ECG Scan Digitization")
st.write("Upload a scanned ECG PDF to extract metadata and digitized lead signals, with diagnostic overlays for review.")
st.warning("Research/engineering tool only. Outputs require qualified review and must not be used as the sole basis for clinical decisions.")
with st.sidebar:
    st.header("Processing settings")
    page_number = st.number_input("PDF page number", min_value=1, value=1, step=1, help="Page numbers start at 1.")
    dpi = st.select_slider("PDF rendering resolution (DPI)", options=[150, 200, 250, 300, 350, 400], value=300)
uploaded_pdf = st.file_uploader("Upload an ECG PDF", type=["pdf"])
if uploaded_pdf is not None:
    st.caption(f"Selected file: {uploaded_pdf.name} ({uploaded_pdf.size:,} bytes)")
    if st.button("Digitize ECG", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing ECG. This may take a little while..."):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp); input_path = root / "uploaded_ecg.pdf"; output_dir = root / "output"; output_dir.mkdir()
                    input_path.write_bytes(uploaded_pdf.getvalue())
                    logs = io.StringIO()
                    with redirect_stdout(logs):
                        bundle = process_pdf(input_path, page_number=int(page_number)-1, dpi=int(dpi), output_dir=output_dir)
                    output = bundle["output"]
                    json_bytes = json.dumps(output, indent=2, ensure_ascii=False).encode("utf-8")
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                        archive.writestr("ecg_data.json", json_bytes)
                        for name, image in bundle["images"].items():
                            ok, encoded = cv2.imencode(".png", image)
                            if ok: archive.writestr(name, encoded.tobytes())
                    st.session_state["ecg_json_bytes"] = json_bytes
                    st.session_state["ecg_zip_bytes"] = zip_buffer.getvalue()
                    st.session_state["ecg_metadata"] = output.get("metadata", {})
                    st.session_state["ecg_leads"] = output.get("leads", {})
                    st.session_state["ecg_rhythm"] = output.get("rhythm_strip")
                    st.session_state["ecg_images"] = {n: cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for n, im in bundle["images"].items()}
                    st.session_state["ecg_log"] = logs.getvalue()
            st.success("ECG digitization completed. Review the outputs below.")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.caption("Check the app logs, confirm the selected PDF page exists, and verify that Tesseract OCR is installed.")
if "ecg_json_bytes" in st.session_state:
    st.divider(); st.subheader("Outputs")
    c1, c2 = st.columns(2)
    with c1: st.download_button("Download digitized ECG JSON", st.session_state["ecg_json_bytes"], "ecg_data.json", "application/json", use_container_width=True)
    with c2: st.download_button("Download JSON + diagnostic images (ZIP)", st.session_state["ecg_zip_bytes"], "ecg_digitization_outputs.zip", "application/zip", use_container_width=True)
    with st.expander("Extracted metadata", expanded=True): st.json(st.session_state["ecg_metadata"])
    with st.expander("Digitized lead data"): st.write(f"Detected lead entries: {len(st.session_state['ecg_leads'])}"); st.json(st.session_state["ecg_leads"])
    with st.expander("Rhythm strip"): st.json(st.session_state["ecg_rhythm"])
    with st.expander("Diagnostic images", expanded=True):
        images = st.session_state.get("ecg_images", {})
        cols = st.columns(min(2, max(1, len(images))))
        for i, (name, image) in enumerate(images.items()):
            with cols[i % len(cols)]: st.markdown(f"**{name}**"); st.image(image, use_container_width=True)
    with st.expander("Processing log"): st.text(st.session_state.get("ecg_log", "No log available."))
st.caption("Privacy: upload only ECG files you are authorized to process. Do not upload identifiable patient data to a third-party deployment without your organization's approval.")
