
from extract_resume import extract_text_from_pdf

pdf_path = "uploads/sample_resume.pdf"

text = extract_text_from_pdf(pdf_path)

print("Extracted Resume Text:")
print(text)
