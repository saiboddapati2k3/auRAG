import requests
from io import BytesIO
from pdfminer.high_level import extract_text

def load_pdf_from_url(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception("Failed to download PDF.")
    pdf_data = BytesIO(response.content)
    text = extract_text(pdf_data)
    return text
