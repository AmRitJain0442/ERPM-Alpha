"""
Setup Script for Gold Standard Data Collection
Helps configure environment and test API connections
"""

import os
import sys

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    else:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")

    required = [
        'requests',
        'pandas',
        'beautifulsoup4',
    ]

    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)

    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False

    return True


def check_directories():
    """Check if required directories exist"""
    print("\nChecking directories...")

    dirs = [
        "data/gold_standard/fred",
        "data/gold_standard/us_census",
        "data/gold_standard/india_commerce"
    ]

    all_exist = True

    for directory in dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} (missing)")
            all_exist = False

    if not all_exist:
        print("\n⚠ Some directories are missing")
        print("\nCreate with:")
        print("  mkdir -p data/gold_standard/fred data/gold_standard/us_census data/gold_standard/india_commerce")
        return False

    return True


def check_api_keys():
    """Check if API keys are configured"""
    print("\nChecking API keys...")

    keys = {
        'FRED_API_KEY': 'Required - FRED data collection will fail without it',
        'CENSUS_API_KEY': 'Optional - Rate limits will be lower without it'
    }

    all_set = True

    for key, description in keys.items():
        value = os.getenv(key)

        if value:
            # Mask the key for security
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✓ {key} = {masked}")
        else:
            print(f"✗ {key} (not set)")
            print(f"  → {description}")
            all_set = False

    if not all_set:
        print("\n⚠ Some API keys are missing")
        print("\nHow to set API keys:")
        print("\n1. Get your API keys:")
        print("   FRED: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("   Census: https://api.census.gov/data/key_signup.html")
        print("\n2. Set as environment variables:")
        print("   Windows:")
        print("     set FRED_API_KEY=your_key")
        print("     set CENSUS_API_KEY=your_key")
        print("\n   Linux/Mac:")
        print("     export FRED_API_KEY=your_key")
        print("     export CENSUS_API_KEY=your_key")
        print("\n3. Or create a .env file (recommended):")
        print("   Copy .env.template to .env and fill in your keys")

    return all_set


def test_api_connection(api_name, test_func):
    """Test API connection"""
    print(f"\nTesting {api_name} connection...")

    try:
        result = test_func()
        if result:
            print(f"✓ {api_name} connection successful")
            return True
        else:
            print(f"✗ {api_name} connection failed")
            return False
    except Exception as e:
        print(f"✗ {api_name} connection error: {e}")
        return False


def test_fred_connection():
    """Test FRED API connection"""
    import requests

    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("  No API key found")
        return False

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': 'DEXINUS',
        'api_key': api_key,
        'file_type': 'json',
        'limit': 1
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'observations' in data:
            return True

    return False


def test_census_connection():
    """Test Census API connection"""
    import requests

    api_key = os.getenv('CENSUS_API_KEY')
    if not api_key:
        print("  No API key found (optional)")
        return True  # It's optional, so return True

    url = "https://api.census.gov/data/timeseries/intltrade/imports/country"
    params = {
        'get': 'CTY_CODE,CTY_NAME',
        'time': '2023',
        'CTY_CODE': '4100',
        'key': api_key
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return True

    return False


def create_env_file():
    """Create .env file from template"""
    if os.path.exists('.env'):
        print("\n.env file already exists")
        return

    if os.path.exists('.env.template'):
        import shutil
        shutil.copy('.env.template', '.env')
        print("\n✓ Created .env file from template")
        print("  → Please edit .env and add your API keys")
    else:
        print("\n✗ .env.template not found")


def main():
    """Main setup function"""
    print("=" * 70)
    print("  Gold Standard Data Collection - Setup")
    print("=" * 70)

    checks = {
        'Python Version': check_python_version(),
        'Dependencies': False,
        'Directories': False,
        'API Keys': False,
    }

    # Only proceed if Python version is OK
    if checks['Python Version']:
        checks['Dependencies'] = check_dependencies()
        checks['Directories'] = check_directories()
        checks['API Keys'] = check_api_keys()

    # Test API connections if keys are set
    if checks['API Keys']:
        print("\n" + "=" * 70)
        print("  Testing API Connections")
        print("=" * 70)

        test_api_connection('FRED', test_fred_connection)
        test_api_connection('Census', test_census_connection)

    # Summary
    print("\n" + "=" * 70)
    print("  Setup Summary")
    print("=" * 70)

    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")

    all_passed = all(checks.values())

    if all_passed:
        print("\n✓ Setup complete! You're ready to collect data.")
        print("\nRun:")
        print("  python collect_all_data.py")
    else:
        print("\n⚠ Setup incomplete. Please address the issues above.")

        # Offer to create .env file
        if not checks['API Keys']:
            create_env_file()

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
