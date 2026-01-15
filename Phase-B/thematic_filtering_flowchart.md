# Thematic Filtering Process - Flowchart

## Main Process Flow

```mermaid
flowchart TD
    Start([Raw GDELT Data<br/>2.5M Articles]) --> Problem{Problem: Naive<br/>Aggregation Fails}
    
    Problem --> |Averages wash out signals| Solution[Solution: Theme-Specific Filtering]
    
    Solution --> Step1[Step 1: Define 4 Relevant Themes]
    
    Step1 --> Economy[Economy Theme<br/>139,953 articles<br/>Keywords: RBI, inflation,<br/>interest rate, GDP]
    Step1 --> Conflict[Conflict Theme<br/>560,938 articles<br/>Keywords: protest, strike,<br/>tension, attack]
    Step1 --> Policy[Policy Theme<br/>594,924 articles<br/>Keywords: government,<br/>legislation, regulation]
    Step1 --> Corporate[Corporate Theme<br/>53,779 articles<br/>Keywords: Adani, Reliance,<br/>Tata, Infosys]
    
    Economy --> Step2[Step 2: Engineer 16 Daily Features]
    Conflict --> Step2
    Policy --> Step2
    Corporate --> Step2
    
    Step2 --> Sentiment[Sentiment Features<br/>5 features<br/>Tone_Economy, Tone_Conflict,<br/>Tone_Policy, Tone_Corporate,<br/>Tone_Overall]
    
    Step2 --> Goldstein[Goldstein Features<br/>2 features<br/>Goldstein_Avg,<br/>Goldstein_Weighted]
    
    Step2 --> Volume[Volume Metrics<br/>6 features<br/>Count_Economy, Count_Conflict,<br/>Count_Policy, Count_Corporate,<br/>Count_Total]
    
    Step2 --> Spikes[Volume Spikes<br/>3 features<br/>Volume_Spike,<br/>Volume_Spike_Economy,<br/>Volume_Spike_Conflict]
    
    Sentiment --> Innovation{Key Innovation:<br/>Goldstein_Weighted}
    Goldstein --> Innovation
    Volume --> Innovation
    Spikes --> Innovation
    
    Innovation --> |Goldstein × NumMentions| Impact[Captures IMPACT<br/>not just sentiment<br/>-10 × 5,000 = -50,000]
    
    Impact --> Step3[Step 3: Correlation Analysis]
    
    Step3 --> Merge[Merge Features with IMF_3<br/>Noise Component]
    
    Merge --> Correlate[Calculate Correlations<br/>merged.corr 'IMF_3']
    
    Correlate --> Results[Results]
    
    Results --> Top1[Tone_Economy: -0.23<br/>Negative sentiment → Volatility]
    Results --> Top2[Goldstein_Weighted: -0.19<br/>Conflict → Instability]
    Results --> Top3[Volume_Spike_Economy: +0.15<br/>News spikes → Volatility]
    Results --> Top4[Count_Corporate: +0.12<br/>Corporate news → Activity]
    
    Top1 --> Insight([Key Insight:<br/>Moderate correlations 0.15-0.25<br/>are GOOD in finance<br/>Perfect correlation would be<br/>arbitraged away])
    Top2 --> Insight
    Top3 --> Insight
    Top4 --> Insight
    
    style Start fill:#e1f5ff
    style Solution fill:#ffe1e1
    style Innovation fill:#fff9e1
    style Insight fill:#e1ffe1
    style Results fill:#f0e1ff
```

## Detailed Theme Logic Flow

```mermaid
flowchart LR
    subgraph Themes["4 Theme Categories"]
        E[Economy<br/>139,953 articles]
        C[Conflict<br/>560,938 articles]
        P[Policy<br/>594,924 articles]
        Co[Corporate<br/>53,779 articles]
    end
    
    subgraph Logic["Business Logic"]
        E --> EL[Central bank policy<br/>affects currency<br/>valuation]
        C --> CL[Geopolitical instability<br/>→ Capital flight<br/>→ Depreciation]
        P --> PL[Regulatory changes<br/>affect investment<br/>flows]
        Co --> CoL[Corporate events<br/>signal economic<br/>health]
    end
    
    subgraph Features["16 Daily Features"]
        EL --> F1[5 Sentiment Scores]
        CL --> F2[2 Goldstein Scores]
        PL --> F3[6 Volume Counts]
        CoL --> F4[3 Volume Spikes]
    end
    
    style E fill:#4CAF50,color:#fff
    style C fill:#F44336,color:#fff
    style P fill:#2196F3,color:#fff
    style Co fill:#FF9800,color:#fff
```

## Feature Engineering Pipeline

```mermaid
flowchart TD
    Raw[Raw Daily Data] --> Group[Group by Date]
    
    Group --> Agg{Aggregation Functions}
    
    Agg --> S1[mean AvgTone<br/>by Theme]
    Agg --> S2[mean GoldsteinScale]
    Agg --> S3[sum GoldsteinScale × NumMentions]
    Agg --> S4[count articles<br/>by Theme]
    Agg --> S5[percent_change<br/>volume]
    
    S1 --> Output[Daily Feature Matrix<br/>362 rows × 16 features]
    S2 --> Output
    S3 --> Output
    S4 --> Output
    S5 --> Output
    
    Output --> Merge[Merge with IMF_3]
    Merge --> Train[Training Dataset]
    
    style Raw fill:#e3f2fd
    style Output fill:#c8e6c9
    style Train fill:#fff9c4
```

## Correlation Analysis Workflow

```mermaid
flowchart LR
    Features[16 Features<br/>Daily Aggregated] --> Merge[Merge on Date]
    IMF3[IMF_3<br/>Noise Component<br/>from VMD] --> Merge
    
    Merge --> Corr[Calculate<br/>Pearson Correlation]
    
    Corr --> Sort[Sort by<br/>Correlation Strength]
    
    Sort --> Top[Top Correlations]
    
    Top --> R1[Tone_Economy: -0.23]
    Top --> R2[Goldstein_Weighted: -0.19]
    Top --> R3[Volume_Spike_Economy: +0.15]
    Top --> R4[Count_Corporate: +0.12]
    
    R1 --> Interpret{Interpretation}
    R2 --> Interpret
    R3 --> Interpret
    R4 --> Interpret
    
    Interpret --> I1[Negative sentiment<br/>predicts volatility]
    Interpret --> I2[Conflict events<br/>create instability]
    Interpret --> I3[News spikes<br/>signal market activity]
    Interpret --> I4[Corporate coverage<br/>reflects market health]
    
    style Features fill:#bbdefb
    style IMF3 fill:#ffccbc
    style Top fill:#c5e1a5
    style Interpret fill:#f8bbd0
```

## Goldstein_Weighted Innovation

```mermaid
flowchart TD
    Start[Article Event] --> Has{Has GoldsteinScale<br/>and NumMentions?}
    
    Has -->|Yes| Calc[Calculate:<br/>Goldstein_Weighted =<br/>GoldsteinScale × NumMentions]
    Has -->|No| Skip[Skip]
    
    Calc --> Example1{Example Scenarios}
    
    Example1 --> Low[Small Event:<br/>Goldstein = -10<br/>Mentions = 5<br/>Weighted = -50<br/>Status: NOISE]
    
    Example1 --> High[Major Crisis:<br/>Goldstein = -10<br/>Mentions = 5,000<br/>Weighted = -50,000<br/>Status: SIGNIFICANT]
    
    Low --> Insight[Captures IMPACT<br/>not just sentiment]
    High --> Insight
    
    Insight --> Daily[Sum all weighted<br/>scores per day]
    
    Daily --> Result[Daily<br/>Goldstein_Weighted<br/>Feature]
    
    style Low fill:#c8e6c9
    style High fill:#ef5350,color:#fff
    style Insight fill:#fff9c4
    style Result fill:#81c784
```

