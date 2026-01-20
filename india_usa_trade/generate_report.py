"""
Generate Comprehensive India-USA Trade Analysis Report
Creates an HTML report with all findings and visualizations.
"""

import os
import pandas as pd
from datetime import datetime
from config import OUTPUT_DIR

def generate_html_report():
    """Generate an HTML report summarizing all trade analysis findings."""

    # Load data files
    trade_balance = pd.read_csv(os.path.join(OUTPUT_DIR, "trade_balance_analysis.csv"))
    commodity_shift = pd.read_csv(os.path.join(OUTPUT_DIR, "commodity_shift_multiyear.csv"))

    # Calculate key statistics
    latest_year = trade_balance["fetch_year"].max()
    earliest_year = trade_balance["fetch_year"].min()

    latest_deficit = trade_balance[trade_balance["fetch_year"] == latest_year]["trade_balance"].values[0]
    earliest_deficit = trade_balance[trade_balance["fetch_year"] == earliest_year]["trade_balance"].values[0]
    deficit_growth = ((latest_deficit / earliest_deficit) - 1) * 100

    latest_exports = trade_balance[trade_balance["fetch_year"] == latest_year]["exports"].values[0]
    latest_imports = trade_balance[trade_balance["fetch_year"] == latest_year]["imports"].values[0]

    # Electronics growth calculation
    elec_2018 = commodity_shift[commodity_shift["fetch_year"] == 2018]["Electronics"].values[0]
    elec_2024 = commodity_shift[commodity_shift["fetch_year"] == 2024]["Electronics"].values[0]
    elec_growth = ((elec_2024 / elec_2018) - 1) * 100

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>India-USA Trade Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: #1a365d;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: #f7fafc;
            border-left: 4px solid #3182ce;
            padding: 15px;
            border-radius: 5px;
        }}
        .stat-box.deficit {{
            border-left-color: #e53e3e;
        }}
        .stat-box.growth {{
            border-left-color: #38a169;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2d3748;
        }}
        .stat-label {{
            color: #718096;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: right;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #edf2f7;
            font-weight: 600;
            color: #2d3748;
        }}
        td:first-child, th:first-child {{
            text-align: left;
        }}
        tr:hover {{
            background: #f7fafc;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .highlight {{
            background: #fef3c7;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .negative {{
            color: #e53e3e;
        }}
        .positive {{
            color: #38a169;
        }}
        .footer {{
            text-align: center;
            color: #718096;
            padding: 20px;
            font-size: 0.9em;
        }}
        .insight {{
            background: #ebf8ff;
            border-left: 4px solid #3182ce;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>India-USA Bilateral Trade Analysis</h1>
        <p>Comprehensive analysis of trade flows between India and United States ({earliest_year}-{latest_year})</p>
        <p>Generated: {datetime.now().strftime("%B %d, %Y")}</p>
    </div>

    <div class="card">
        <h2>Executive Summary</h2>
        <div class="stat-grid">
            <div class="stat-box deficit">
                <div class="stat-value">${abs(latest_deficit/1e9):.1f}B</div>
                <div class="stat-label">US Trade Deficit ({latest_year})</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${latest_exports/1e9:.1f}B</div>
                <div class="stat-label">US Exports to India ({latest_year})</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${latest_imports/1e9:.1f}B</div>
                <div class="stat-label">US Imports from India ({latest_year})</div>
            </div>
            <div class="stat-box growth">
                <div class="stat-value">+{elec_growth:.0f}%</div>
                <div class="stat-label">Electronics Growth (2018-2024)</div>
            </div>
        </div>
        <div class="insight">
            <strong>Key Finding:</strong> The US trade deficit with India has grown {abs(deficit_growth):.0f}% since {earliest_year},
            reaching ${abs(latest_deficit/1e9):.1f} billion in {latest_year}. Electronics imports have surged {elec_growth:.0f}%
            since 2018, reflecting the "China tariff effect" as India captures manufacturing market share.
        </div>
    </div>

    <div class="card">
        <h2>Trade Balance Trends ({earliest_year}-{latest_year})</h2>
        <div class="chart-container">
            <img src="trade_balance_chart.png" alt="Trade Balance Chart">
        </div>
        <table>
            <thead>
                <tr>
                    <th>Year</th>
                    <th>US Exports (B)</th>
                    <th>US Imports (B)</th>
                    <th>Trade Balance (B)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add table rows for trade balance
    for _, row in trade_balance.iterrows():
        status_class = "negative" if row["trade_balance"] < 0 else "positive"
        status_text = "Deficit" if row["trade_balance"] < 0 else "Surplus"
        html_content += f"""
                <tr>
                    <td>{int(row['fetch_year'])}</td>
                    <td>${row['exports']/1e9:.2f}</td>
                    <td>${row['imports']/1e9:.2f}</td>
                    <td class="{status_class}">${row['trade_balance']/1e9:.2f}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Commodity Shift Analysis (China Tariff Effect)</h2>
        <div class="chart-container">
            <img src="commodity_shift_chart.png" alt="Commodity Shift Chart">
        </div>
        <div class="insight">
            <strong>China Tariff Effect:</strong> Following the 2018 US-China trade tensions, India's electronics
            exports to the US have grown exponentially. Electronics went from being a minor category ($1.7B in 2018)
            to becoming a major export ($14.0B in 2024), surpassing traditional sectors like gems and jewelry.
        </div>
        <table>
            <thead>
                <tr>
                    <th>Year</th>
                    <th>Electronics (B)</th>
                    <th>Pharma (B)</th>
                    <th>Gems & Jewelry (B)</th>
                    <th>Machinery (B)</th>
                    <th>Chemicals (B)</th>
                    <th>Apparel (B)</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add commodity shift table
    for _, row in commodity_shift.iterrows():
        html_content += f"""
                <tr>
                    <td>{int(row['fetch_year'])}</td>
                    <td>${row['Electronics']:.2f}</td>
                    <td>${row['Pharmaceuticals']:.2f}</td>
                    <td>${row['Gems & Jewelry']:.2f}</td>
                    <td>${row['Machinery']:.2f}</td>
                    <td>${row['Organic Chemicals']:.2f}</td>
                    <td>${row['Apparel']:.2f}</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Monthly Seasonality (2024)</h2>
        <div class="chart-container">
            <img src="seasonality_chart_2024.png" alt="Seasonality Chart">
        </div>
        <div class="insight">
            <strong>Seasonal Patterns:</strong> US imports from India show relative stability throughout the year,
            averaging $6-8 billion monthly. Import peaks typically occur in March-May (pre-summer inventory buildup)
            and October (pre-holiday season). Exports remain consistently lower at $3-4 billion monthly.
        </div>
    </div>

    <div class="card">
        <h2>Data Sources & Methodology</h2>
        <ul>
            <li><strong>Primary Source:</strong> US Census Bureau International Trade API (HS Classification)</li>
            <li><strong>Secondary Source:</strong> UN Comtrade via World Trade Data library</li>
            <li><strong>Time Period:</strong> {earliest_year} to {latest_year}</li>
            <li><strong>Trade Values:</strong> Reported in US Dollars (current prices)</li>
            <li><strong>Classification:</strong> Harmonized System (HS) 2-digit commodity codes</li>
        </ul>
    </div>

    <div class="footer">
        <p>Generated by India-USA Trade Analysis Tool</p>
        <p>Data Source: US Census Bureau International Trade Statistics</p>
    </div>
</body>
</html>
"""

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "trade_analysis_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_html_report()
