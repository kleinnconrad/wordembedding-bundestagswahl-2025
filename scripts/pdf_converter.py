import pdfplumber
import os

# 1. Absolute Path Setup
# This ensures it works regardless of where you call it from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PDF_DIR = os.path.join(PROJECT_ROOT, "corpora")
TXT_DIR = os.path.join(PROJECT_ROOT, "data")

def convert_pdfs_to_txt():
    # Ensure the output directory exists
    if not os.path.exists(TXT_DIR):
        print(f"Creating directory: {TXT_DIR}")
        os.makedirs(TXT_DIR)

    parties = ["cdu", "gruene", "spd", "afd"]
    
    print(f"Looking for PDFs in: {PDF_DIR}")

    for party in parties:
        pdf_path = os.path.join(PDF_DIR, f"{party}.pdf")
        txt_path = os.path.join(TXT_DIR, f"{party}.txt")

        if not os.path.exists(pdf_path):
            print(f"❌ Skipping {party.upper()}: File NOT found at {pdf_path}")
            continue

        print(f"➔ Processing {party.upper()}...")
        
        char_count = 0
        with pdfplumber.open(pdf_path) as pdf:
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                for i, page in enumerate(pdf.pages):
                    # Define a bounding box (x0, top, x1, bottom)
                    # We take 10% off top and bottom to avoid headers/footers
                    bbox = (0, page.height * 0.1, page.width, page.height * 0.9)
                    cropped_page = page.crop(bbox)
                    
                    text = cropped_page.extract_text()
                    
                    if text:
                        clean_text = text.replace("-\n", "")
                        txt_file.write(clean_text + "\n")
                        char_count += len(clean_text)
                    
        if char_count > 0:
            print(f"   ✅ Success: Extracted {char_count} characters to {party}.txt")
        else:
            print(f"   ⚠️ Warning: No text was extracted from {party}.pdf. (Check if it's a scanned image PDF)")

if __name__ == "__main__":
    convert_pdfs_to_txt()