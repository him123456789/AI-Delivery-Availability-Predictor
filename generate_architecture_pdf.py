#!/usr/bin/env python3
"""
Architecture Document PDF Generator
===================================
Creates a professional PDF from the architecture document with enhanced styling for technical diagrams.
"""

import os
from datetime import datetime

def create_architecture_html():
    """Convert architecture markdown to styled HTML"""
    
    with open('ARCHITECTURE_DOCUMENT.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Enhanced markdown to HTML conversion for architecture content
    html_lines = []
    in_code_block = False
    in_yaml_block = False
    
    for line in content.split('\n'):
        if line.startswith('```yaml'):
            html_lines.append('<pre class="yaml-block">')
            in_yaml_block = True
        elif line.startswith('```python'):
            html_lines.append('<pre class="python-block">')
            in_code_block = True
        elif line.startswith('```hcl'):
            html_lines.append('<pre class="hcl-block">')
            in_code_block = True
        elif line.startswith('```') and (in_code_block or in_yaml_block):
            html_lines.append('</pre>')
            in_code_block = False
            in_yaml_block = False
        elif in_code_block or in_yaml_block:
            html_lines.append(line)
        elif line.startswith('# '):
            html_lines.append(f'<h1 class="main-heading">{line[2:]}</h1>')
        elif line.startswith('## '):
            html_lines.append(f'<h2 class="section-heading">{line[3:]}</h2>')
        elif line.startswith('### '):
            html_lines.append(f'<h3 class="subsection-heading">{line[4:]}</h3>')
        elif line.startswith('#### '):
            html_lines.append(f'<h4 class="component-heading">{line[5:]}</h4>')
        elif line.startswith('- ') or line.startswith('* '):
            html_lines.append(f'<li class="list-item">{line[2:]}</li>')
        elif line.strip() == '':
            html_lines.append('<br>')
        elif line.strip().startswith('┌') or line.strip().startswith('│') or line.strip().startswith('└'):
            # ASCII diagrams
            html_lines.append(f'<div class="ascii-diagram">{line}</div>')
        else:
            # Handle bold, italic, and code formatting
            processed_line = line
            processed_line = processed_line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
            processed_line = processed_line.replace('*', '<em>', 1).replace('*', '</em>', 1)
            processed_line = processed_line.replace('`', '<code>', 1).replace('`', '</code>', 1)
            html_lines.append(f'<p class="body-text">{processed_line}</p>')
    
    return '\n'.join(html_lines)

def create_architecture_pdf_html():
    """Create complete HTML document for architecture PDF"""
    
    content_html = create_architecture_html()
    
    html_document = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI Delivery Availability Prediction System - Architecture Document</title>
        <style>
            @page {{
                size: A4;
                margin: 0.8in;
                @top-center {{
                    content: "AI Delivery Availability Prediction System - Architecture Document";
                    font-size: 9pt;
                    color: #666;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 5px;
                }}
                @bottom-center {{
                    content: "Page " counter(page) " | Architecture v1.0";
                    font-size: 9pt;
                    color: #666;
                    border-top: 1px solid #ddd;
                    padding-top: 5px;
                }}
            }}
            
            body {{
                font-family: 'Segoe UI', 'Arial', sans-serif;
                line-height: 1.5;
                color: #2d3748;
                font-size: 10pt;
                margin: 0;
                padding: 0;
            }}
            
            .cover-page {{
                text-align: center;
                padding-top: 180px;
                page-break-after: always;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin: -0.8in;
                padding: 2in 0.8in;
                min-height: 100vh;
            }}
            
            .cover-title {{
                font-size: 32pt;
                font-weight: bold;
                margin-bottom: 20px;
                line-height: 1.1;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            
            .cover-subtitle {{
                font-size: 18pt;
                margin-bottom: 40px;
                opacity: 0.9;
            }}
            
            .cover-info {{
                font-size: 12pt;
                margin-top: 80px;
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 10px;
                backdrop-filter: blur(10px);
            }}
            
            .main-heading {{
                color: #2b6cb0;
                font-size: 20pt;
                font-weight: bold;
                border-bottom: 3px solid #2b6cb0;
                padding-bottom: 8px;
                margin-top: 30px;
                margin-bottom: 15px;
                page-break-before: always;
            }}
            
            .section-heading {{
                color: #2c5282;
                font-size: 16pt;
                font-weight: bold;
                margin-top: 25px;
                margin-bottom: 12px;
                border-left: 4px solid #2b6cb0;
                padding-left: 12px;
                background-color: #f7fafc;
                padding: 8px 12px;
            }}
            
            .subsection-heading {{
                color: #2d3748;
                font-size: 13pt;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 8px;
            }}
            
            .component-heading {{
                color: #4a5568;
                font-size: 11pt;
                font-weight: bold;
                margin-top: 15px;
                margin-bottom: 6px;
            }}
            
            .body-text {{
                text-align: justify;
                margin-bottom: 8px;
                line-height: 1.6;
            }}
            
            .list-item {{
                margin-bottom: 4px;
                padding-left: 5px;
            }}
            
            .yaml-block {{
                background: linear-gradient(135deg, #e6fffa 0%, #f0fff4 100%);
                border: 1px solid #38b2ac;
                border-left: 4px solid #38b2ac;
                border-radius: 5px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                line-height: 1.4;
                margin: 10px 0;
                overflow-x: auto;
            }}
            
            .python-block {{
                background: linear-gradient(135deg, #fef5e7 0%, #fefcbf 100%);
                border: 1px solid #d69e2e;
                border-left: 4px solid #d69e2e;
                border-radius: 5px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                line-height: 1.4;
                margin: 10px 0;
                overflow-x: auto;
            }}
            
            .hcl-block {{
                background: linear-gradient(135deg, #e6f3ff 0%, #f0f8ff 100%);
                border: 1px solid #3182ce;
                border-left: 4px solid #3182ce;
                border-radius: 5px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                line-height: 1.4;
                margin: 10px 0;
                overflow-x: auto;
            }}
            
            .ascii-diagram {{
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                text-align: center;
                line-height: 1.2;
                color: #495057;
            }}
            
            code {{
                background-color: #f1f3f4;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                color: #d63384;
            }}
            
            strong {{
                font-weight: bold;
                color: #2d3748;
            }}
            
            em {{
                font-style: italic;
                color: #4a5568;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 9pt;
            }}
            
            th, td {{
                border: 1px solid #e2e8f0;
                padding: 8px;
                text-align: left;
            }}
            
            th {{
                background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
                font-weight: bold;
                color: #2d3748;
            }}
            
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            
            .architecture-highlight {{
                background: linear-gradient(135deg, #bee3f8 0%, #90cdf4 100%);
                border: 1px solid #3182ce;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
            }}
            
            .decision-record {{
                background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
                border: 1px solid #e53e3e;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
            }}
            
            .toc {{
                page-break-after: always;
                margin-bottom: 30px;
            }}
            
            .toc h2 {{
                color: #2b6cb0;
                border-bottom: 2px solid #2b6cb0;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            
            .toc ul {{
                list-style-type: none;
                padding-left: 0;
                line-height: 1.8;
            }}
            
            .toc li {{
                margin-bottom: 6px;
                padding-left: 20px;
                border-left: 2px solid #e2e8f0;
            }}
            
            .toc a {{
                text-decoration: none;
                color: #2d3748;
                font-weight: 500;
            }}
            
            .toc a:hover {{
                color: #2b6cb0;
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
                System Architecture Document
            </div>
            <div class="cover-info">
                <p><strong>Version:</strong> 1.0</p>
                <p><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
                <p><strong>Project:</strong> NextGen Fastest and Smartest Delivery Network</p>
                <p><strong>Document Type:</strong> Technical Architecture</p>
                <p><strong>Classification:</strong> Internal Technical Documentation</p>
                <p><strong>Status:</strong> Final</p>
            </div>
        </div>
        
        <!-- Table of Contents -->
        <div class="toc">
            <h2>📋 Table of Contents</h2>
            <ul>
                <li><a href="#architecture-overview">1. Architecture Overview</a></li>
                <li><a href="#system-components">2. System Components</a></li>
                <li><a href="#data-architecture">3. Data Architecture</a></li>
                <li><a href="#ml-pipeline-architecture">4. ML Pipeline Architecture</a></li>
                <li><a href="#api-architecture">5. API Architecture</a></li>
                <li><a href="#infrastructure-architecture">6. Infrastructure Architecture</a></li>
                <li><a href="#security-architecture">7. Security Architecture</a></li>
                <li><a href="#scalability--performance">8. Scalability & Performance</a></li>
                <li><a href="#deployment-architecture">9. Deployment Architecture</a></li>
                <li><a href="#integration-patterns">10. Integration Patterns</a></li>
            </ul>
        </div>
        
        <!-- Document Content -->
        <div class="content">
            {content_html}
        </div>
        
        <!-- Document Summary -->
        <div style="page-break-before: always; margin-top: 30px;">
            <h2>📊 Architecture Summary</h2>
            <div class="architecture-highlight">
                <h3>Key Architectural Decisions</h3>
                <ul>
                    <li><strong>Microservices Architecture:</strong> Independent, scalable services</li>
                    <li><strong>Event-Driven Design:</strong> Asynchronous communication patterns</li>
                    <li><strong>Cloud-Native:</strong> Containerized deployment on AWS/Kubernetes</li>
                    <li><strong>MLOps Pipeline:</strong> Automated model training and deployment</li>
                    <li><strong>API-First:</strong> RESTful services with OpenAPI specifications</li>
                </ul>
            </div>
            
            <table style="margin-top: 20px;">
                <tr><th>Component</th><th>Technology</th><th>Purpose</th></tr>
                <tr><td>Prediction Service</td><td>Python, FastAPI, scikit-learn</td><td>Real-time ML predictions</td></tr>
                <tr><td>Customer Service</td><td>Python, FastAPI, PostgreSQL</td><td>Customer data management</td></tr>
                <tr><td>Calendar Service</td><td>Python, FastAPI, Redis</td><td>External calendar integration</td></tr>
                <tr><td>Analytics Service</td><td>Python, Spark, ClickHouse</td><td>Business intelligence</td></tr>
                <tr><td>API Gateway</td><td>AWS API Gateway/Kong</td><td>Request routing and security</td></tr>
                <tr><td>Message Queue</td><td>Apache Kafka/RabbitMQ</td><td>Async communication</td></tr>
                <tr><td>Container Platform</td><td>Kubernetes/ECS</td><td>Container orchestration</td></tr>
                <tr><td>Monitoring</td><td>Prometheus, Grafana, ELK</td><td>Observability stack</td></tr>
            </table>
        </div>
    </body>
    </html>
    """
    
    return html_document

def main():
    """Generate architecture PDF"""
    print("🏗️  AI Delivery Prediction System - Architecture PDF Generator")
    print("=" * 70)
    
    if not os.path.exists('ARCHITECTURE_DOCUMENT.md'):
        print("❌ ARCHITECTURE_DOCUMENT.md not found!")
        return
    
    print("📋 Creating HTML version of the architecture document...")
    html_content = create_architecture_pdf_html()
    
    # Save HTML file
    html_filename = 'AI_Delivery_Architecture_Document.html'
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    html_size = os.path.getsize(html_filename)
    print(f"✅ HTML document created: {html_filename}")
    print(f"📄 File size: {html_size:,} bytes ({html_size/1024:.1f} KB)")
    
    print(f"\n🔄 Converting HTML to PDF...")
    pdf_filename = 'AI_Delivery_Architecture_Document.pdf'
    
    # Try to convert to PDF
    conversion_commands = [
        f'wkhtmltopdf --page-size A4 --margin-top 0.8in --margin-bottom 0.8in --margin-left 0.8in --margin-right 0.8in --enable-local-file-access {html_filename} {pdf_filename}',
        f'weasyprint {html_filename} {pdf_filename}',
        f'chromium-browser --headless --disable-gpu --print-to-pdf={pdf_filename} --print-to-pdf-no-header {html_filename}',
    ]
    
    pdf_created = False
    for cmd in conversion_commands:
        try:
            result = os.system(cmd + ' 2>/dev/null')
            if result == 0 and os.path.exists(pdf_filename):
                pdf_size = os.path.getsize(pdf_filename)
                print(f"✅ PDF created successfully: {pdf_filename}")
                print(f"📄 File size: {pdf_size:,} bytes ({pdf_size/1024/1024:.2f} MB)")
                pdf_created = True
                break
        except:
            continue
    
    if not pdf_created:
        print(f"⚠️  PDF conversion failed, but HTML version is available!")
        print(f"💡 You can open {html_filename} in a browser and print to PDF")
    
    print(f"\n🎉 Architecture Document Generation Complete!")
    print(f"📍 Location: {os.path.abspath('.')}")
    print(f"📊 Files created:")
    print(f"   • HTML: ✅ {html_filename}")
    print(f"   • PDF:  {'✅' if pdf_created else '❌'} {pdf_filename if pdf_created else 'Not created'}")
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
