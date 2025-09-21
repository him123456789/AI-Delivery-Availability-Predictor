#!/usr/bin/env python3
"""
Simple PDF Generator for Design Document
========================================
Creates a professional PDF from the markdown design document using available libraries.
"""

import os
from datetime import datetime

def create_pdf_with_weasyprint():
    """Create PDF using WeasyPrint"""
    try:
        import weasyprint
        import markdown
        
        print("📄 Creating PDF using WeasyPrint...")
        
        # Read markdown content
        with open('DESIGN_DOCUMENT.md', 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content, extensions=['tables', 'codehilite', 'toc'])
        
        # Create styled HTML
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>AI Delivery Availability Prediction System - Design Document</title>
            <style>
                @page {{
                    size: A4;
                    margin: 1in;
                    @top-center {{
                        content: "AI Delivery Availability Prediction System - Design Document";
                        font-size: 10pt;
                        color: #666;
                    }}
                    @bottom-center {{
                        content: "Page " counter(page);
                        font-size: 10pt;
                        color: #666;
                    }}
                }}
                
                body {{
                    font-family: 'Arial', 'Helvetica', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    font-size: 11pt;
                }}
                
                .cover-page {{
                    text-align: center;
                    padding-top: 200px;
                    page-break-after: always;
                }}
                
                .cover-title {{
                    font-size: 32pt;
                    font-weight: bold;
                    color: #1f77b4;
                    margin-bottom: 30px;
                    line-height: 1.2;
                }}
                
                .cover-subtitle {{
                    font-size: 16pt;
                    color: #666;
                    margin-bottom: 50px;
                }}
                
                .cover-info {{
                    font-size: 12pt;
                    color: #333;
                    margin-top: 100px;
                }}
                
                h1 {{
                    color: #1f77b4;
                    font-size: 20pt;
                    border-bottom: 3px solid #1f77b4;
                    padding-bottom: 10px;
                    margin-top: 30px;
                    page-break-before: always;
                }}
                
                h2 {{
                    color: #2c5282;
                    font-size: 16pt;
                    margin-top: 25px;
                    border-left: 4px solid #1f77b4;
                    padding-left: 15px;
                }}
                
                h3 {{
                    color: #2d3748;
                    font-size: 14pt;
                    margin-top: 20px;
                }}
                
                h4 {{
                    color: #4a5568;
                    font-size: 12pt;
                    margin-top: 15px;
                }}
                
                p {{
                    text-align: justify;
                    margin-bottom: 10px;
                }}
                
                ul, ol {{
                    margin-bottom: 12px;
                    padding-left: 20px;
                }}
                
                li {{
                    margin-bottom: 4px;
                }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    font-size: 10pt;
                }}
                
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                
                th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                    color: #2d3748;
                }}
                
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                
                code {{
                    background-color: #f1f3f4;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                    font-size: 9pt;
                }}
                
                pre {{
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 5px;
                    padding: 10px;
                    overflow-x: auto;
                    font-family: 'Courier New', monospace;
                    font-size: 9pt;
                    line-height: 1.4;
                }}
                
                blockquote {{
                    border-left: 4px solid #1f77b4;
                    margin: 15px 0;
                    padding-left: 15px;
                    color: #666;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <!-- Cover Page -->
            <div class="cover-page">
                <div class="cover-title">
                    AI Delivery Availability<br>
                    Prediction System
                </div>
                <div class="cover-subtitle">
                    Technical Design Document
                </div>
                <div class="cover-info">
                    <p><strong>Version:</strong> 1.0</p>
                    <p><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
                    <p><strong>Project:</strong> NextGen Fastest and Smartest Delivery Network</p>
                    <p><strong>Classification:</strong> Technical Documentation</p>
                </div>
            </div>
            
            <!-- Document Content -->
            {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        filename = 'AI_Delivery_Prediction_Design_Document.pdf'
        weasyprint.HTML(string=styled_html).write_pdf(filename)
        
        file_size = os.path.getsize(filename)
        print(f"✅ PDF created successfully: {filename}")
        print(f"📄 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        return filename
        
    except Exception as e:
        print(f"❌ WeasyPrint failed: {str(e)}")
        return create_pdf_with_reportlab()

def create_pdf_with_reportlab():
    """Create PDF using ReportLab"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        
        print("📄 Creating PDF using ReportLab...")
        
        # Read markdown content
        with open('DESIGN_DOCUMENT.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create PDF document
        filename = 'AI_Delivery_Prediction_Design_Document.pdf'
        doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
        
        # Define styles
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1f77b4'),
            spaceBefore=20,
            spaceAfter=10
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        
        # Build document
        story = []
        
        # Cover page
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("AI Delivery Availability Prediction System", title_style))
        story.append(Paragraph("Technical Design Document", styles['Heading2']))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(f"Version: 1.0", body_style))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", body_style))
        story.append(Paragraph("Project: NextGen Fastest and Smartest Delivery Network", body_style))
        story.append(PageBreak())
        
        # Process content
        sections = content.split('\n## ')
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            lines = section.split('\n')
            title = lines[0].strip()
            
            story.append(Paragraph(f"{i}. {title}", heading1_style))
            
            for line in lines[1:]:
                if line.strip():
                    clean_line = line.replace('**', '').replace('*', '').replace('`', '')
                    if len(clean_line) > 0:
                        story.append(Paragraph(clean_line, body_style))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(story)
        
        file_size = os.path.getsize(filename)
        print(f"✅ PDF created successfully: {filename}")
        print(f"📄 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        return filename
        
    except Exception as e:
        print(f"❌ ReportLab failed: {str(e)}")
        return create_pdf_with_fpdf()

def create_pdf_with_fpdf():
    """Create PDF using FPDF"""
    try:
        from fpdf import FPDF
        
        print("📄 Creating PDF using FPDF...")
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'AI Delivery Availability Prediction System - Design Document', 0, 1, 'C')
                self.ln(5)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        # Read content
        with open('DESIGN_DOCUMENT.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        pdf = PDF()
        pdf.add_page()
        
        # Cover page
        pdf.set_font('Arial', 'B', 20)
        pdf.ln(50)
        pdf.cell(0, 15, 'AI Delivery Availability', 0, 1, 'C')
        pdf.cell(0, 15, 'Prediction System', 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font('Arial', '', 14)
        pdf.cell(0, 10, 'Technical Design Document', 0, 1, 'C')
        pdf.ln(20)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f'Version: 1.0', 0, 1, 'C')
        pdf.cell(0, 8, f'Date: {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
        pdf.add_page()
        
        # Content
        pdf.set_font('Arial', '', 10)
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('# '):
                pdf.set_font('Arial', 'B', 16)
                pdf.ln(10)
                pdf.cell(0, 8, line[2:], 0, 1)
                pdf.set_font('Arial', '', 10)
            elif line.startswith('## '):
                pdf.set_font('Arial', 'B', 14)
                pdf.ln(8)
                pdf.cell(0, 7, line[3:], 0, 1)
                pdf.set_font('Arial', '', 10)
            elif line.startswith('### '):
                pdf.set_font('Arial', 'B', 12)
                pdf.ln(6)
                pdf.cell(0, 6, line[4:], 0, 1)
                pdf.set_font('Arial', '', 10)
            elif line.strip():
                # Handle text wrapping
                if len(line) > 90:
                    words = line.split(' ')
                    current_line = ''
                    for word in words:
                        if len(current_line + word) < 90:
                            current_line += word + ' '
                        else:
                            if current_line:
                                pdf.cell(0, 5, current_line.strip(), 0, 1)
                            current_line = word + ' '
                    if current_line:
                        pdf.cell(0, 5, current_line.strip(), 0, 1)
                else:
                    pdf.cell(0, 5, line, 0, 1)
            else:
                pdf.ln(2)
        
        filename = 'AI_Delivery_Prediction_Design_Document.pdf'
        pdf.output(filename)
        
        file_size = os.path.getsize(filename)
        print(f"✅ PDF created successfully: {filename}")
        print(f"📄 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        return filename
        
    except Exception as e:
        print(f"❌ FPDF failed: {str(e)}")
        return None

def main():
    """Main function"""
    print("🚀 AI Delivery Prediction System - PDF Generator")
    print("=" * 60)
    
    if not os.path.exists('DESIGN_DOCUMENT.md'):
        print("❌ DESIGN_DOCUMENT.md not found!")
        return
    
    print("📋 Attempting to create PDF using available libraries...")
    
    # Try different methods in order of preference
    filename = create_pdf_with_weasyprint()
    
    if filename and os.path.exists(filename):
        print(f"\n🎉 SUCCESS!")
        print(f"📍 PDF Location: {os.path.abspath(filename)}")
        print(f"📅 Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show file details
        stat = os.stat(filename)
        print(f"📊 File Details:")
        print(f"   • Size: {stat.st_size:,} bytes ({stat.st_size/1024/1024:.2f} MB)")
        print(f"   • Created: {datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}")
        
    else:
        print("❌ Failed to create PDF with all available methods")
        print("💡 Please ensure you have the required dependencies installed:")
        print("   pip install weasyprint reportlab fpdf2 markdown")

if __name__ == "__main__":
    main()
