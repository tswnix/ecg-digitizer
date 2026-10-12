# ECG Scan Digitization

A Streamlit interface for a Python ECG digitization pipeline. Upload a scanned ECG PDF, process a selected page, inspect diagnostic overlays, and download structured output.

## Features

- Render a selected PDF page at a configurable DPI.
- Detect orientation, waveform rows, lead separators, and lead regions.
- Use OCR and layout rules to identify leads and extract available header/footer metadata.
- Digitize waveform traces using ECG calibration information.
- Resample lead signals to 500 Hz.
- Download structured JSON and a ZIP containing JSON plus diagnostic images.

## Important limitations

This is an engineering/research tool, not a medical device or diagnostic system. Scan quality, ECG layout, OCR quality, calibration marks, page rotation, and image artifacts can affect results. Review diagnostic images and independently validate extracted signals. Do not use these outputs as the sole basis for clinical decisions. Do not upload identifiable patient data to a third-party service without explicit organizational approval.

## Files

```text
ecg-digitizer/
├── app.py
├── ecg_layout_detection.py
├── requirements.txt
├── packages.txt
├── README.md
├── GITHUB_DESCRIPTION.txt
└── .gitignore
```

## Run locally on Windows

1. Install Python and Tesseract OCR. If the ECG metadata is Spanish, install Spanish language data too.
2. In PowerShell, from this folder, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If Tesseract is not on PATH, set `TESSERACT_CMD` to the full executable path in the same PowerShell session, for example:

```powershell
$env:TESSERACT_CMD = "C:\Path\To\Tesseract-OCR\tesseract.exe"
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository. For company code, use a private repository and follow your organization's source-code and data policies.
2. Upload all files from this folder to the repository root.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Click **Create app** and select the repository, branch (usually `main`), and `app.py`.
5. Choose a Python version compatible with your local environment and dependencies in **Advanced settings**.
6. Deploy, then review logs if the build fails. Confirm the app's visibility/access settings before sharing its URL.

`requirements.txt` lists Python dependencies. `packages.txt` installs the Tesseract OCR executable and Spanish language data on the Linux host.

## Outputs

### `ecg_data.json`

Includes extracted metadata, digitized leads, sampling rate, start time, duration, amplitude samples in mV, rhythm-strip data when detected, source page, DPI, orientation correction, paper speed, gain, and per-row calibration details.

### Diagnostic images

- `ecg_lead_regions.png`: detected lead-region boundaries and labels.
- `ecg_lead_separators.png`: waveform rows and separator positions.
- `ecg_digitized_overlay.png`: digitized traces overlaid on the ECG image.

## Troubleshooting

- **Tesseract not found:** verify Tesseract is installed; locally set `TESSERACT_CMD` if it is not on PATH. On Community Cloud, confirm `packages.txt` is at repository root.
- **Spanish OCR language missing:** install Spanish language data locally; Community Cloud installs `tesseract-ocr-spa`.
- **Processing fails:** confirm the selected PDF page exists, try a lower DPI if memory is limited, and inspect Community Cloud logs.

## License

No license has been specified. Do not assume this project is open source or redistribute it without authorization from the code owner or your organization.
