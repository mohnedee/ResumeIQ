
from reportlab.pdfgen import canvas

pdf_path = "uploads/sample_resume.pdf"

pdf = canvas.Canvas(pdf_path)

pdf.drawString(100, 750, "Name: Alex Smith")
pdf.drawString(100, 720, "Skills: Python, SQL, Machine Learning")
pdf.drawString(100, 690, "Education: Computer Engineering")
pdf.drawString(100, 660, "Experience: Junior Software Developer")

pdf.save()

print(f"Sample resume created at: {pdf_path}")
