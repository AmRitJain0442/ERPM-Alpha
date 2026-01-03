# India Ministry of Commerce Trade Data

## Data Source
https://tradestat.commerce.gov.in/

## Manual Download Instructions

1. Visit the Trade Statistics portal: https://tradestat.commerce.gov.in/meidb/default.asp

2. Navigate to "Import/Export Data Bank"

3. Select parameters:
   - Period: Monthly
   - Year: Select year (2010 onwards)
   - Country: United States of America
   - Trade Type: Both (Exports & Imports)

4. Download the data as CSV or Excel format

5. Save the file in this directory with naming convention:
   - manual_download_[YEAR].csv
   - Example: manual_download_2020.csv

6. Run the script to process and combine all downloaded files

## Automated Collection

The API-based collection is being developed. Currently, manual download is the most reliable method.

## Data Fields

Expected fields in the downloaded data:
- Period (Year-Month)
- HS Code / Product Category
- Export Value (USD)
- Import Value (USD)
- Quantity
- Unit

## Notes

- Indian data may differ slightly from US Census data due to:
  - Different reporting periods
  - Currency conversion timing
  - Classification differences
  - Reporting lags
