from pathlib import Path
import shutil
import subprocess
import sys


BROWSER_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def find_browser() -> Path | None:
    for path in BROWSER_CANDIDATES:
        if path.exists():
            return path
    for name in ("chrome", "msedge"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    return None


def main() -> None:
    root = Path(__file__).resolve().parent
    html_path = root / "poster_presentation.html"
    pdf_path = root / "poster_presentation_final.pdf"
    preview_path = root / "poster_preview_final.png"

    browser = find_browser()
    if browser is None:
        raise SystemExit("No Chrome/Edge executable found for headless PDF rendering.")

    for output in (pdf_path, preview_path):
        if output.exists():
            output.unlink()

    pdf_cmd = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    screenshot_cmd = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--window-size=820,1160",
        f"--screenshot={preview_path}",
        html_path.as_uri(),
    ]

    subprocess.run(pdf_cmd, check=True)
    subprocess.run(screenshot_cmd, check=True)

    pdf_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    preview_size_kb = preview_path.stat().st_size / 1024
    print(f"Created {pdf_path.name} ({pdf_size_mb:.2f} MB)")
    print(f"Created {preview_path.name} ({preview_size_kb:.0f} KB)")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Browser render failed with exit code {exc.returncode}", file=sys.stderr)
        raise
