
### B1 Candidate Algorithms
We looked at three model families. Gradient‑Boosted Decision Trees (GBDT) add many small decision trees together so each one fixes the mistakes of the last. Long Short‑Term Memory (LSTM) networks are sequence models that can remember patterns across time. A spatio‑temporal graph model treats locations as nodes and lets information move across space and time. Each model builds in different assumptions about how data behaves (a hypothesis space), which changes bias and variance (Russell & Norvig, 2020).

### B2 Selected Algorithm and Justification
We chose GBDT for the baseline. It optimizes squared error, which matches the RMSE metric we report, and it often balances underfitting and overfitting well on tabular data (Russell & Norvig, 2020). Two strengths are strong performance with non‑linear tabular features and little need for scaling. Two limits are weak handling of long‑range time patterns without hand‑made lags and sensitivity to data shifts that may need re‑tuning. This follows the textbook idea: start with a simple, well‑regularized model matched to the loss, then improve if errors show leftover structure (Russell & Norvig, 2020).

### C Implementation (short)
We load the Excel file, add calendar features and one‑step lags for key pollutants, split by time (first 80% train, last 20% test), train a histogram‑based GBDT regressor, and save metrics, permutation feature importance, and plots. Everything runs from src/train_and_eval.py.

### D1 Evaluation Metrics
We use RMSE because it matches the squared‑error objective used in training. We add MAPE to show relative error in percentages so results are easy to compare across value ranges. This follows guidance to align metrics with the learning goal and include a scale‑aware view (Russell & Norvig, 2020).

### D2 Results
On the test set, PM2.5 has RMSE 5.1418433169 and MAPE 14.2004813911. HealthRiskScore has RMSE 0.1795709693 and MAPE 1.4374191354. The artifacts folder also includes permutation‑based feature‑importance CSVs and actual‑versus‑predicted plots to help diagnose model behavior.

### D3 Analysis of Metrics
PM2.5’s higher MAPE means bigger percentage errors during spikes and fast changes. A single lag likely misses longer time effects. HealthRiskScore’s much lower MAPE shows the model fits that target more smoothly with smaller relative error. In textbook terms, our current hypothesis space has acceptable bias and variance for HealthRiskScore but leaves structure in PM2.5 that we can still model (Russell & Norvig, 2020).

### D4 Areas for Improvement
Next steps include richer time features (more lags, rolling means, trend flags), sequence models like LSTMs for longer memory, optional spatial modeling with a graph if location data exists, and tuned or stacked ensembles to adjust bias and variance. This matches the iterative modeling loop: study errors, adjust the hypothesis space, and re‑evaluate with the same metrics (Russell & Norvig, 2020).

### E. Sources
Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Pearson.

