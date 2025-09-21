#!/usr/bin/env python3
"""
PDF Generator for AI Delivery Availability Prediction System Design Document
============================================================================

This script converts the markdown design document to a professional PDF format
with proper formatting, table of contents, and styling.
"""

import markdown
import pdfkit
from datetime import datetime
import os
import re

def create_pdf_from_markdown():
    """Convert the design document markdown to PDF"""
    
    # Read the markdown file
    with open('DESIGN_DOCUMENT.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Create HTML content with enhanced styling
    html_content = f"""
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
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 10pt;
                    color: #666;
                }}
            }}
            
            body {{
                font-family: 'Arial', 'Helvetica', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
                margin: 0;
                padding: 0;
            }}
            
            .cover-page {{
                text-align: center;
                padding-top: 200px;
                page-break-after: always;
            }}
            
            .cover-title {{
                font-size: 36pt;
                font-weight: bold;
                color: #1f77b4;
                margin-bottom: 30px;
                line-height: 1.2;
            }}
            
            .cover-subtitle {{
                font-size: 18pt;
                color: #666;
                margin-bottom: 50px;
            }}
            
            .cover-info {{
                font-size: 14pt;
                color: #333;
                margin-top: 100px;
            }}
            
            h1 {{
                color: #1f77b4;
                font-size: 24pt;
                border-bottom: 3px solid #1f77b4;
                padding-bottom: 10px;
                margin-top: 40px;
                page-break-before: always;
            }}
            
            h2 {{
                color: #2c5282;
                font-size: 18pt;
                margin-top: 30px;
                border-left: 4px solid #1f77b4;
                padding-left: 15px;
            }}
            
            h3 {{
                color: #2d3748;
                font-size: 14pt;
                margin-top: 25px;
            }}
            
            h4 {{
                color: #4a5568;
                font-size: 12pt;
                margin-top: 20px;
            }}
            
            p {{
                text-align: justify;
                margin-bottom: 12px;
            }}
            
            ul, ol {{
                margin-bottom: 15px;
                padding-left: 25px;
            }}
            
            li {{
                margin-bottom: 5px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 11pt;
            }}
            
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
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
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }}
            
            pre {{
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 5px;
                padding: 15px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                line-height: 1.4;
            }}
            
            blockquote {{
                border-left: 4px solid #1f77b4;
                margin: 20px 0;
                padding-left: 20px;
                color: #666;
                font-style: italic;
            }}
            
            .toc {{
                page-break-after: always;
                margin-bottom: 30px;
            }}
            
            .toc h2 {{
                color: #1f77b4;
                border-bottom: 2px solid #1f77b4;
                padding-bottom: 10px;
            }}
            
            .toc ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            
            .toc li {{
                margin-bottom: 8px;
                padding-left: 20px;
            }}
            
            .toc a {{
                text-decoration: none;
                color: #333;
            }}
            
            .architecture-diagram {{
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                text-align: center;
            }}
            
            .metric-box {{
                background-color: #e3f2fd;
                border: 1px solid #1f77b4;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
            }}
            
            .status-success {{
                color: #28a745;
                font-weight: bold;
            }}
            
            .status-warning {{
                color: #ffc107;
                font-weight: bold;
            }}
            
            .status-danger {{
                color: #dc3545;
                font-weight: bold;
            }}
            
            .page-break {{
                page-break-before: always;
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
                <p><strong>Document Type:</strong> Technical Architecture & Design</p>
            </div>
        </div>
        
        <!-- Table of Contents -->
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#executive-summary">1. Executive Summary</a></li>
                <li><a href="#system-architecture">2. System Architecture</a></li>
                <li><a href="#data-model">3. Data Model</a></li>
                <li><a href="#machine-learning-model">4. Machine Learning Model</a></li>
                <li><a href="#api-design">5. API Design</a></li>
                <li><a href="#user-interface">6. User Interface</a></li>
                <li><a href="#performance-specifications">7. Performance Specifications</a></li>
                <li><a href="#security--privacy">8. Security & Privacy</a></li>
                <li><a href="#deployment-architecture">9. Deployment Architecture</a></li>
                <li><a href="#testing-strategy">10. Testing Strategy</a></li>
                <li><a href="#monitoring--maintenance">11. Monitoring & Maintenance</a></li>
                <li><a href="#future-enhancements">12. Future Enhancements</a></li>
            </ul>
        </div>
        
        <!-- Document Content -->
        {markdown.markdown(markdown_content, extensions=['tables', 'codehilite', 'toc'])}
        
        <!-- Document Footer -->
        <div class="page-break">
            <h2>Document Information</h2>
            <table>
                <tr><th>Field</th><th>Value</th></tr>
                <tr><td>Document Title</td><td>AI Delivery Availability Prediction System - Design Document</td></tr>
                <tr><td>Version</td><td>1.0</td></tr>
                <tr><td>Creation Date</td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                <tr><td>Authors</td><td>AI Development Team</td></tr>
                <tr><td>Classification</td><td>Technical Documentation</td></tr>
                <tr><td>Status</td><td>Final</td></tr>
            </table>
        </div>
    </body>
    </html>
    """
    
    # Configure PDF options
    options = {
        'page-size': 'A4',
        'margin-top': '1in',
        'margin-right': '0.8in',
        'margin-bottom': '1in',
        'margin-left': '0.8in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None,
        'print-media-type': None,
        'disable-smart-shrinking': None,
        'header-spacing': 5,
        'footer-spacing': 5,
        'header-font-size': 8,
        'footer-font-size': 8,
    }
    
    # Generate PDF
    output_filename = 'AI_Delivery_Prediction_Design_Document.pdf'
    
    try:
        pdfkit.from_string(html_content, output_filename, options=options)
        print(f"✅ PDF generated successfully: {output_filename}")
        print(f"📄 File size: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")
        return output_filename
    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        print("💡 Trying alternative method...")
        return create_pdf_alternative_method()

def create_pdf_alternative_method():
    """Alternative PDF generation using reportlab"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
        
        # Read markdown content
        with open('DESIGN_DOCUMENT.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create PDF document
        filename = 'AI_Delivery_Prediction_Design_Document.pdf'
        doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles
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
            fontSize=18,
            textColor=colors.HexColor('#1f77b4'),
            spaceBefore=20,
            spaceAfter=12
        )
        
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c5282'),
            spaceBefore=15,
            spaceAfter=8
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        
        # Build document content
        story = []
        
        # Cover page
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("AI Delivery Availability<br/>Prediction System", title_style))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Technical Design Document", styles['Heading2']))
        story.append(Spacer(1, 1*inch))
        
        # Document info
        doc_info = [
            ['Version:', '1.0'],
            ['Date:', datetime.now().strftime('%B %d, %Y')],
            ['Project:', 'NextGen Fastest and Smartest Delivery Network'],
            ['Document Type:', 'Technical Architecture & Design']
        ]
        
        info_table = Table(doc_info, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        story.append(info_table)
        story.append(PageBreak())
        
        # Process markdown content sections
        sections = content.split('\n## ')
        
        for i, section in enumerate(sections):
            if i == 0:
                continue  # Skip the title section
            
            lines = section.split('\n')
            section_title = lines[0].strip()
            section_content = '\n'.join(lines[1:])
            
            # Add section title
            story.append(Paragraph(f"{i}. {section_title}", heading1_style))
            
            # Process section content
            paragraphs = section_content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # Clean up markdown formatting
                    clean_para = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', para)
                    clean_para = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_para)
                    clean_para = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', clean_para)
                    
                    if clean_para.startswith('### '):
                        story.append(Paragraph(clean_para[4:], heading2_style))
                    elif clean_para.startswith('- ') or clean_para.startswith('* '):
                        story.append(Paragraph(f"• {clean_para[2:]}", body_style))
                    else:
                        story.append(Paragraph(clean_para, body_style))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF generated successfully using ReportLab: {filename}")
        print(f"📄 File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
        return filename
        
    except ImportError:
        print("❌ ReportLab not available. Installing required packages...")
        return create_simple_pdf()

def create_simple_pdf():
    """Simple PDF creation using weasyprint or basic HTML to PDF"""
    try:
        import weasyprint
        
        # Read markdown and convert to HTML
        with open('DESIGN_DOCUMENT.md', 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>AI Delivery Availability Prediction System - Design Document</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                h1 {{ color: #1f77b4; border-bottom: 2px solid #1f77b4; }}
                h2 {{ color: #2c5282; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
                code {{ background-color: #f1f3f4; padding: 2px 4px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div style="text-align: center; margin-bottom: 50px;">
                <h1 style="font-size: 28px; color: #1f77b4;">AI Delivery Availability Prediction System</h1>
                <h2 style="color: #666;">Technical Design Document</h2>
                <p><strong>Version:</strong> 1.0 | <strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            {markdown.markdown(markdown_content, extensions=['tables', 'codehilite'])}
        </body>
        </html>
        """
        
        # Generate PDF
        filename = 'AI_Delivery_Prediction_Design_Document.pdf'
        weasyprint.HTML(string=html_content).write_pdf(filename)
        
        print(f"✅ PDF generated successfully using WeasyPrint: {filename}")
        print(f"📄 File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
        return filename
        
    except ImportError:
        print("❌ WeasyPrint not available. Creating basic text-based PDF...")
        return create_text_pdf()

def create_text_pdf():
    """Create a basic PDF with text content"""
    from fpdf import FPDF
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'AI Delivery Availability Prediction System - Design Document', 0, 1, 'C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    # Read markdown content
    with open('DESIGN_DOCUMENT.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    # Process content line by line
    lines = content.split('\n')
    for line in lines:
        if line.startswith('# '):
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, line[2:], 0, 1)
            pdf.ln(5)
            pdf.set_font('Arial', '', 12)
        elif line.startswith('## '):
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, line[3:], 0, 1)
            pdf.ln(3)
            pdf.set_font('Arial', '', 12)
        elif line.startswith('### '):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, line[4:], 0, 1)
            pdf.ln(2)
            pdf.set_font('Arial', '', 12)
        elif line.strip():
            # Handle long lines
            if len(line) > 80:
                words = line.split(' ')
                current_line = ''
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + ' '
                    else:
                        if current_line:
                            pdf.cell(0, 6, current_line.strip(), 0, 1)
                        current_line = word + ' '
                if current_line:
                    pdf.cell(0, 6, current_line.strip(), 0, 1)
            else:
                pdf.cell(0, 6, line, 0, 1)
        else:
            pdf.ln(3)
    
    filename = 'AI_Delivery_Prediction_Design_Document.pdf'
    pdf.output(filename)
    
    print(f"✅ Basic PDF generated successfully: {filename}")
    print(f"📄 File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
    return filename

def install_pdf_dependencies():
    """Install required packages for PDF generation"""
    import subprocess
    import sys
    
    packages = [
        'markdown',
        'pdfkit', 
        'weasyprint',
        'reportlab',
        'fpdf2'
    ]
    
    print("📦 Installing PDF generation dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}")
    
    # Install wkhtmltopdf for pdfkit
    print("📦 Note: For best results, install wkhtmltopdf system package:")
    print("   Ubuntu/Debian: sudo apt-get install wkhtmltopdf")
    print("   CentOS/RHEL: sudo yum install wkhtmltopdf")
    print("   macOS: brew install wkhtmltopdf")

def main():
    """Main function to generate PDF"""
    print("🚀 Starting PDF generation for Design Document...")
    print("=" * 60)
    
    # Check if markdown file exists
    if not os.path.exists('DESIGN_DOCUMENT.md'):
        print("❌ DESIGN_DOCUMENT.md not found!")
        return
    
    try:
        # Try different PDF generation methods
        filename = create_pdf_from_markdown()
        
        if filename and os.path.exists(filename):
            print(f"\n🎉 SUCCESS! PDF created: {filename}")
            print(f"📍 Location: {os.path.abspath(filename)}")
            
            # Display file info
            file_size = os.path.getsize(filename)
            print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            print(f"📅 Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        else:
            print("❌ Failed to generate PDF with all methods")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Try installing dependencies with: pip install markdown pdfkit weasyprint reportlab fpdf2")

if __name__ == "__main__":
    main()
