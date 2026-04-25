from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_investor_report(path: str, metrics: dict) -> None:
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(72, height - 80, "ATLAS Investor Report")
    
    # Subtitle / Date
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(72, height - 100, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Separator Line
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(72, height - 110, width - 72, height - 110)

    # Content
    c.setFillColorRGB(0, 0, 0)
    y = height - 150
    
    for key, value in metrics.items():
        # Key
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, str(key))
        
        # Value
        c.setFont("Helvetica", 12)
        # Align values at a specific x-coordinate for a clean look
        c.drawString(250, y, str(value))
        
        y -= 30
        
        # Add a light line between rows
        c.setStrokeColorRGB(0.9, 0.9, 0.9)
        c.line(72, y + 10, width - 72, y + 10)
        
        if y < 100:
            c.showPage()
            y = height - 80
            c.setFont("Helvetica", 12)

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(72, 50, "ATLAS Multi-Agent Startup Management Simulation")

    c.save()
