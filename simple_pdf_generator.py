#!/usr/bin/env python3
"""
Simple PDF Generator using built-in libraries
==============================================
Creates a PDF from the design document using only standard Python libraries.
"""

import os
import sys
from datetime import datetime

def create_html_from_markdown():
    """Convert markdown to HTML manually"""
    
    with open('DESIGN_DOCUMENT.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple markdown to HTML conversion
    html_lines = []
    in_code_block = False
    
    for line in content.split('\n'):
        if line.startswith('```'):
            if in_code_block:
                html_lines.append('</pre>')
                in_code_block = False
            else:
                html_lines.append('<pre>')
                in_code_block = True
        elif in_code_block:
            html_lines.append(line)
        elif line.startswith('# '):
            html_lines.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('#### '):
            html_lines.append(f'<h4>{line[5:]}</h4>')
        elif line.startswith('- ') or line.startswith('* '):
            html_lines.append(f'<li>{line[2:]}</li>')
        elif line.strip() == '':
            html_lines.append('<br>')
        else:
            # Handle bold and italic
            processed_line = line
            processed_line = processed_line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
            processed_line = processed_line.replace('*', '<em>', 1).replace('*', '</em>', 1)
            processed_line = processed_line.replace('`', '<code>', 1).replace('`', '</code>', 1)
            html_lines.append(f'<p>{processed_line}</p>')
    
    return '\n'.join(html_lines)

def create_styled_html():
    """Create a complete HTML document with styling"""
    
    content_html = create_html_from_markdown()
    
    html_document = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI Delivery Availability Prediction System - Design Document</title>
        <style>
            @page {{
                size: A4;
                margin: 1in;
            }}
            
            body {{
                font-family: Arial, Helvetica, sans-serif;
                line-height: 1.6;
                color: #333;
                font-size: 11pt;
                max-width: 100%;
            }}
            
            .cover-page {{
                text-align: center;
                padding-top: 150px;
                page-break-after: always;
                margin-bottom: 50px;
            }}
            
            .cover-title {{
                font-size: 28pt;
                font-weight: bold;
                color: #1f77b4;
                margin-bottom: 20px;
                line-height: 1.2;
            }}
            
            .cover-subtitle {{
                font-size: 16pt;
                color: #666;
                margin-bottom: 40px;
            }}
            
            .cover-info {{
                font-size: 12pt;
                color: #333;
                margin-top: 80px;
            }}
            
            h1 {{
                color: #1f77b4;
                font-size: 18pt;
                border-bottom: 2px solid #1f77b4;
                padding-bottom: 8px;
                margin-top: 30px;
                page-break-before: auto;
            }}
            
            h2 {{
                color: #2c5282;
                font-size: 14pt;
                margin-top: 25px;
                border-left: 3px solid #1f77b4;
                padding-left: 10px;
            }}
            
            h3 {{
                color: #2d3748;
                font-size: 12pt;
                margin-top: 20px;
            }}
            
            h4 {{
                color: #4a5568;
                font-size: 11pt;
                margin-top: 15px;
            }}
            
            p {{
                text-align: justify;
                margin-bottom: 8px;
            }}
            
            li {{
                margin-bottom: 4px;
            }}
            
            pre {{
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                overflow-x: auto;
            }}
            
            code {{
                background-color: #f1f3f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
            }}
            
            strong {{
                font-weight: bold;
                color: #2d3748;
            }}
            
            em {{
                font-style: italic;
                color: #4a5568;
            }}
            
            .header {{
                position: fixed;
                top: 0.5in;
                left: 1in;
                right: 1in;
                text-align: center;
                font-size: 9pt;
                color: #666;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            
            .footer {{
                position: fixed;
                bottom: 0.5in;
                left: 1in;
                right: 1in;
                text-align: center;
                font-size: 9pt;
                color: #666;
                border-top: 1px solid #ddd;
                padding-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            AI Delivery Availability Prediction System - Design Document
        </div>
        
        <div class="footer">
            Page 1 | Generated on {datetime.now().strftime('%B %d, %Y')}
        </div>
        
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
                <p><strong>Status:</strong> Final</p>
            </div>
        </div>
        
        <!-- Document Content -->
        <div class="content">
            {content_html}
        </div>
        
        <!-- Document Footer -->
        <div style="page-break-before: always; margin-top: 50px;">
            <h2>Document Information</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border: 1px solid #ddd;">
                    <td style="border: 1px solid #ddd; padding: 8px; background-color: #f8f9fa;"><strong>Field</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px; background-color: #f8f9fa;"><strong>Value</strong></td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Document Title</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">AI Delivery Availability Prediction System - Design Document</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Version</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">1.0</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Creation Date</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Authors</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">AI Development Team</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Classification</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Technical Documentation</td>
                </tr>
            </table>
        </div>
    </body>
    </html>
    """
    
    return html_document

def save_html_file():
    """Save the HTML version of the document"""
    html_content = create_styled_html()
    
    filename = 'AI_Delivery_Prediction_Design_Document.html'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    file_size = os.path.getsize(filename)
    print(f"✅ HTML document created: {filename}")
    print(f"📄 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    return filename

def convert_html_to_pdf():
    """Try to convert HTML to PDF using available system tools"""
    html_file = 'AI_Delivery_Prediction_Design_Document.html'
    pdf_file = 'AI_Delivery_Prediction_Design_Document.pdf'
    
    # Try different system commands for HTML to PDF conversion
    commands = [
        f'wkhtmltopdf --page-size A4 --margin-top 1in --margin-bottom 1in --margin-left 1in --margin-right 1in {html_file} {pdf_file}',
        f'weasyprint {html_file} {pdf_file}',
        f'prince {html_file} -o {pdf_file}',
        f'chromium-browser --headless --disable-gpu --print-to-pdf={pdf_file} {html_file}',
        f'google-chrome --headless --disable-gpu --print-to-pdf={pdf_file} {html_file}'
    ]
    
    for cmd in commands:
        try:
            result = os.system(cmd + ' 2>/dev/null')
            if result == 0 and os.path.exists(pdf_file):
                file_size = os.path.getsize(pdf_file)
                print(f"✅ PDF created successfully: {pdf_file}")
                print(f"📄 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                return pdf_file
        except:
            continue
    
    return None

def main():
    """Main function"""
    print("🚀 AI Delivery Prediction System - Simple PDF Generator")
    print("=" * 65)
    
    if not os.path.exists('DESIGN_DOCUMENT.md'):
        print("❌ DESIGN_DOCUMENT.md not found!")
        return
    
    print("📋 Creating HTML version of the design document...")
    html_file = save_html_file()
    
    print(f"\n📄 HTML document ready: {html_file}")
    print(f"📍 Location: {os.path.abspath(html_file)}")
    
    print(f"\n🔄 Attempting to convert HTML to PDF...")
    pdf_file = convert_html_to_pdf()
    
    if pdf_file:
        print(f"\n🎉 SUCCESS! PDF created: {pdf_file}")
        print(f"📍 Location: {os.path.abspath(pdf_file)}")
    else:
        print(f"\n⚠️  PDF conversion failed, but HTML version is available!")
        print(f"💡 You can:")
        print(f"   1. Open {html_file} in a browser and print to PDF")
        print(f"   2. Install wkhtmltopdf: sudo apt-get install wkhtmltopdf")
        print(f"   3. Use online HTML to PDF converters")
    
    print(f"\n📊 Summary:")
    print(f"   • HTML Document: ✅ {html_file}")
    print(f"   • PDF Document: {'✅' if pdf_file else '❌'} {pdf_file or 'Not created'}")
    print(f"   • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
