"""
Simple Python script to convert Markdown to HTML then use browser automation
This will create a PDF with all images intact
"""
import markdown
from pathlib import Path
import webbrowser
import os
import time

def convert_markdown_to_html(md_file_path, html_output_path):
    """
    Convert a markdown file to a standalone HTML file with embedded CSS
    
    Args:
        md_file_path: Path to the markdown file
        html_output_path: Path for the output HTML file
    """
    # Read the markdown file
    print(f"Reading markdown file: {md_file_path}")
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML with extensions
    print("Converting markdown to HTML...")
    html_body = markdown.markdown(
        md_content,
        extensions=[
            'tables',           # Support for tables
            'fenced_code',      # Support for code blocks
            'codehilite',       # Syntax highlighting
            'toc',              # Table of contents
            'nl2br',            # Newline to break
        ]
    )
    
    # Add CSS styling for better output
    css_style = """
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
            background: white;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 30px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 25px;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }
        pre {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 20px 0;
        }
        pre code {
            background-color: transparent;
            color: #ecf0f1;
            padding: 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 25px 0;
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
            margin: 25px auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
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
            margin: 40px 0;
        }
        @media print {
            body {
                padding: 20px;
            }
            h1, h2, h3 {
                page-break-after: avoid;
            }
            img, pre, table {
                page-break-inside: avoid;
            }
        }
    </style>
    """
    
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
        {html_body}
    </body>
    </html>
    """
    
    # Write HTML file
    print(f"Writing HTML file: {html_output_path}")
    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(html_document)
    
    print(f"✓ HTML file created successfully!")
    return html_output_path

def print_pdf_instructions(html_file_path):
    """Print instructions for converting HTML to PDF"""
    print("\n" + "="*70)
    print("HTML file created successfully!")
    print("="*70)
    print(f"\nHTML file location: {html_file_path}")
    print("\nTo convert to PDF:")
    print("\nOption 1: Use your browser (Recommended)")
    print("-" * 40)
    print("1. Open the HTML file in your browser (Chrome/Edge recommended)")
    print("2. Press Ctrl+P (or Cmd+P on Mac)")
    print("3. Select 'Save as PDF' as the printer")
    print("4. Adjust settings:")
    print("   - Layout: Portrait")
    print("   - Paper size: A4")
    print("   - Margins: Default or Custom")
    print("   - Options: Enable 'Background graphics'")
    print("5. Click 'Save' and choose location")
    print("\nOption 2: Use the Markdown PDF extension in VS Code")
    print("-" * 40)
    print("1. Open RESEARCH_PRESENTATION.md in VS Code")
    print("2. Press Ctrl+Shift+P")
    print("3. Type 'Markdown PDF: Export (pdf)'")
    print("4. Press Enter")
    print("\nThe HTML version will automatically open in your browser...")

if __name__ == "__main__":
    # Define file paths
    script_dir = Path(__file__).parent
    md_file = script_dir / "RESEARCH_PRESENTATION.md"
    html_file = script_dir / "RESEARCH_PRESENTATION.html"
    
    try:
        # Convert to HTML
        html_path = convert_markdown_to_html(str(md_file), str(html_file))
        
        # Print instructions
        print_pdf_instructions(html_path)
        
        # Open in browser
        print("\nOpening HTML file in your default browser...")
        time.sleep(1)
        webbrowser.open('file://' + os.path.abspath(html_path))
        
        print("\n✓ Done! You can now print to PDF from your browser.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
