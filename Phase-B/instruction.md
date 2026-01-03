I will proceed with the strategy to align your GDELT data with the "Noise" (IMF 3) component you just extracted.

Step B: The GDELT Alignment Strategy
Now that you have the Red Line (IMF 3 - The Noise), we need to engineer your GDELT data so it actually correlates with this line.

Most people fail here because they treat news as a "daily average." Markets don't react to averages; they react to spikes and specific themes.

We are going to build a "Theme-Specific Impact Score" instead of a generic Goldstein score.

1. The Concept: "Thematic Filtering"
The INR/USD rate doesn't care if there is a political protest about water shortages. It does care if there is a protest about:

RBI (Central Bank) Policy

Adani/Reliance (Major Conglomerates)

Import/Export Taxes

US Fed Rates

We need to filter your GDELT data for these specific themes.

2. The Engineering Strategy
We will create a new DataFrame for your news data with the following columns (features):

Tone_Economy: Average Tone of articles mentioning "Economy", "Inflation", "RBI", "Tax".

Tone_Conflict: Average Tone of articles mentioning "Protest", "Strike", "Geopolitical".

Goldstein_Weighted: The Goldstein score weighted by NumMentions. (A Goldstein score of -10 in an article mentioned 5 times is irrelevant. A score of -10 in an article mentioned 5000 times is a market crash).

Volume_Spike: The count of articles today vs yesterday. (High volume = High Volatility, regardless of sentiment).