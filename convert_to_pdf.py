"""
Convert RESEARCH_PRESENTATION.md to PDF with all images intact
"""
import markdown
import pdfkit
import os
from pathlib import Path

def convert_markdown_to_pdf(md_file_path, output_pdf_path):
    """
    Convert a markdown file to PDF with images
    
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
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 25px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 20px;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }
        pre {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
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
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            border: 1px solid #ddd;
            padding: 10px;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
    </style>
    """
    
    # Create complete HTML document
    html_document = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>USD/INR Exchange Rate Volatility Prediction</title>
        {css_style}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Configure pdfkit options
    options = {
        'page-size': 'A4',
        'margin-top': '20mm',
        'margin-right': '20mm',
        'margin-bottom': '20mm',
        'margin-left': '20mm',
        'encoding': "UTF-8",
        'enable-local-file-access': None,  # Allow loading local images
        'no-outline': None,
        'print-media-type': None,
        'dpi': 300,  # High resolution
    }
    
    # Get the directory of the markdown file to resolve relative image paths
    md_dir = os.path.dirname(os.path.abspath(md_file_path))
    
    print(f"Converting HTML to PDF...")
    print(f"Working directory: {md_dir}")
    
    # Convert HTML to PDF
    # Use configuration to handle relative image paths
    pdfkit.from_string(
        html_document, 
        output_pdf_path, 
        options=options,
        configuration=None  # Will use system wkhtmltopdf
    )
    
    print(f"✓ PDF created successfully: {output_pdf_path}")
    print(f"File size: {os.path.getsize(output_pdf_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # Define file paths
    script_dir = Path(__file__).parent
    md_file = script_dir / "RESEARCH_PRESENTATION.md"
    pdf_file = script_dir / "RESEARCH_PRESENTATION.pdf"
    
    try:
        convert_markdown_to_pdf(str(md_file), str(pdf_file))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure wkhtmltopdf is installed:")
        print("Download from: https://wkhtmltopdf.org/downloads.html")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
