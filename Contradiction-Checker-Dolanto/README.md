Two ways to run
Google Colab (GPU)Local web app (CPU)Use forFull / large documentsSmall test PDFs (1–10 pages)Speed~25–30 sec/pageMinutes/page (CPU is slow)SetupSee "Setup — Colab" belowSee "Setup — Local web app" below

Setup — Local web app (review_app.py)
The web app lets you upload a PDF, run the pipeline, and review the extracted
clauses and tables in a browser. It runs on CPU, so use small test PDFs.

1. Install Python
Python 3.10, 3.11 or 3.12 (not 3.13).

2. Get the files
Put review_app.py, dvs_extract_paddleocrvl.py and requirements.txt in one
folder. The app imports the pipeline, so both .py files must sit together.

3. Create a virtual environment
bash python -m venv venv
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

4. Install dependencies
bash pip install -r requirements.txt

5. Run the app
bash streamlit run review_app.py

Open the URL it prints (usually http://localhost:8501). Upload a small PDF,
choose a page range, and click Run pipeline.

First run downloads the PaddleOCR-VL model (~2 GB) once. The app shows a
"Loading OCR engine" stage while this happens — it can take 10+ minutes the
first time, then the model is cached.
