import fitz

pdf_path = r"c:\Users\danie\Documents\Projects\CI\CI\paper\Paper 1--ADynamicFuzzy Rule and Attribute Management framework for FIS.pdf"
doc = fitz.open(pdf_path)

with open(r"c:\Users\danie\Documents\Projects\CI\CI\paper\extracted_text.txt", "w", encoding="utf-8") as f:
    for page in doc:
        f.write(page.get_text())
