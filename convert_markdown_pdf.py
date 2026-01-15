"""
Convert RESEARCH_PRESENTATION.md to PDF with images using markdown-pdf
This uses Chromium/Chrome to render the PDF
"""
import subprocess
import sys
from pathlib import Path

def check_and_install_markdown_pdf():
    """Check if markdown-pdf is installed globally via npm"""
    try:
        result = subprocess.run(
            ['markdown-pdf', '--version'],
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            print(f"✓ markdown-pdf is installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("markdown-pdf is not installed.")
    print("\nTo install markdown-pdf, you need Node.js and npm.")
    print("Then run: npm install -g markdown-pdf")
    return False

def convert_with_markdown_pdf(md_file, pdf_file):
    """Convert markdown to PDF using markdown-pdf CLI tool"""
    print(f"\nConverting {md_file} to PDF...")
    
    result = subprocess.run(
        ['markdown-pdf', str(md_file), '-o', str(pdf_file)],
        capture_output=True,
        text=True,
        shell=True
    )
    
    if result.returncode == 0:
        print(f"✓ PDF created successfully: {pdf_file}")
        import os
        file_size_mb = os.path.getsize(pdf_file) / (1024*1024)
        print(f"File size: {file_size_mb:.2f} MB")
        return True
    else:
        print(f"Error: {result.stderr}")
        return False

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    md_file = script_dir / "RESEARCH_PRESENTATION.md"
    pdf_file = script_dir / "RESEARCH_PRESENTATION.pdf"
    
    if check_and_install_markdown_pdf():
        convert_with_markdown_pdf(md_file, pdf_file)
    else:
        print("\n" + "="*60)
        print("ALTERNATIVE: Use an online converter or install Node.js")
        print("="*60)
        print("\nOption 1: Install Node.js and markdown-pdf")
        print("  1. Download Node.js from https://nodejs.org/")
        print("  2. Run: npm install -g markdown-pdf")
        print("  3. Run this script again")
        print("\nOption 2: Use VS Code extension")
        print("  1. Install 'Markdown PDF' extension in VS Code")
        print("  2. Open RESEARCH_PRESENTATION.md")
        print("  3. Press Ctrl+Shift+P and type 'Markdown PDF: Export (pdf)'")
