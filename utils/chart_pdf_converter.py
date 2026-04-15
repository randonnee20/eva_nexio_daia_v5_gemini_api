"""차트 PDF 변환기"""

class ChartPDFConverter:
    def __init__(self):
        pass
    
    def convert_to_pdf(self, chart_paths, output_path):
        """차트들을 PDF로 변환"""
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from PIL import Image
        
        c = canvas.Canvas(str(output_path), pagesize=A4)
        width, height = A4
        
        for chart_path in chart_paths:
            img = Image.open(chart_path)
            c.drawImage(str(chart_path), 50, 50, 
                       width-100, height-100, 
                       preserveAspectRatio=True)
            c.showPage()
        
        c.save()