"""
Convert RESEARCH_PRESENTATION.md to PDF with all images intact using WeasyPrint
"""
import markdown
from weasyprint import HTML, CSS
import os
from pathlib import Path

def convert_markdown_to_pdf(md_file_path, output_pdf_path):
    """
    Convert a markdown file to PDF with images using WeasyPrint
    
    Args:
        md_file_path: Path to the markdown file
        output_pdf_path: Path for the output PDF file
    """
    # Read the markdown file
    print(f"Reading markdown file: {md_file_path}")
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML with extensions
    print("Converting markdown to HTML...")
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'tables',           # Support for tables
            'fenced_code',      # Support for code blocks
            'codehilite',       # Syntax highlighting
            'toc',              # Table of contents
            'nl2br',            # Newline to break
        ]
    )
    
    # Add CSS styling for better PDF output
    css_style = """
    <style>
        @page {
            size: A4;
            margin: 2cm;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            font-size: 11pt;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
            font-size: 24pt;
            page-break-before: auto;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 25px;
            font-size: 18pt;
            page-break-after: avoid;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 20px;
            font-size: 14pt;
            page-break-after: avoid;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 9pt;
        }
        pre {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            page-break-inside: avoid;
        }
        pre code {
            background-color: transparent;
            color: #ecf0f1;
            padding: 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            page-break-inside: avoid;
            font-size: 10pt;
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 10px;
            text-align: left;
        }
        td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            page-break-inside: avoid;
        }
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }
        hr {
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }
        .page-break {
            page-break-before: always;
        }
    </style>
    """
    
    # Get the directory of the markdown file to resolve relative image paths
    md_dir = os.path.dirname(os.path.abspath(md_file_path))
    
    # Create complete HTML document
    html_document = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Predicting USD/INR Exchange Rate Volatility Using News Sentiment Analysis</title>
        {css_style}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    print(f"Converting HTML to PDF...")
    print(f"Working directory: {md_dir}")
    
    # Convert HTML to PDF using WeasyPrint
    # base_url is important for resolving relative image paths
    HTML(string=html_document, base_url=md_dir).write_pdf(
        output_pdf_path,
        stylesheets=None,
        zoom=1,
        attachments=None
    )
    
    print(f"✓ PDF created successfully: {output_pdf_path}")
    file_size_mb = os.path.getsize(output_pdf_path) / (1024*1024)
    print(f"File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    # Define file paths
    script_dir = Path(__file__).parent
    md_file = script_dir / "RESEARCH_PRESENTATION.md"
    pdf_file = script_dir / "RESEARCH_PRESENTATION.pdf"
    
    try:
        convert_markdown_to_pdf(str(md_file), str(pdf_file))
        print("\n" + "="*60)
        print("SUCCESS! Your PDF has been created.")
        print("="*60)
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
