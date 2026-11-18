

\# PathSnap — Role B Progress Status



\*\*Responsible:\*\* Jose  

\*\*Role:\*\* Models and Bayesian Hyperparameter Tuning  

\*\*Branch:\*\* b/models  

\*\*Status:\*\* Implementation in progress



\## 1. Completed Work



\### Environment Setup

\- Python virtual environment configured.

\- Installed and verified:

&#x20; - pandas

&#x20; - NumPy

&#x20; - scikit-learn

&#x20; - PyArrow

&#x20; - Matplotlib

&#x20; - Seaborn

&#x20; - Optuna

&#x20; - Joblib



\### Dataset Integration

\- Integrated candidate training and student-test Parquet files.

\- Integrated observation-level reference label CSV files.

\- Implemented dataset loading and validation.

\- Implemented automatic selection of `f\_\*` feature columns.

\- Added checks for missing values, invalid labels and feature consistency.



\### Machine Learning Models

Implemented five classifiers:



1\. Logistic Regression

2\. K-Nearest Neighbors

3\. Random Forest

4\. HistGradientBoosting

5\. Multi-Layer Perceptron



\### Model Evaluation Framework

\- Implemented 5-fold GroupKFold cross-validation.

\- Grouping is performed by `trip\_id`.

\- Implemented Macro F1 evaluation.

\- Implemented precision and recall.

\- Implemented Log Loss.

\- Implemented observation-level Top-1 accuracy.

\- Implemented candidate-miss rate calculation.

\- Implemented training and inference time measurement.

\- Implemented comparison plots.



\### Bayesian Hyperparameter Optimisation

\- Integrated Optuna.

\- Implemented Tree-structured Parzen Estimator (TPE).

\- Implemented GroupKFold-based optimisation objective.

\- Added configurable trial count.

\- Added persistent SQLite study storage.

\- Added trial-history CSV export.

\- Added best-parameter JSON export.



\### Git Integration

\- Created branch `b/models`.

\- Added model training and tuning scripts.

\- Merged the latest `main` branch into `b/models`.

\- Added `.gitignore` rules for Python virtual environments and cache files.



\## 2. Dataset Information



Actual dataset inspection confirmed:



| Dataset property | Training | Student Test |

|---|---:|---:|

| Candidate rows | 1,869,591 | 209,581 |

| Trips | 80 | 10 |

| Features | 15 | 15 |

| Reference observations | 208,417 | 19,160 |

| Missing values | 0 | 0 |



Training label distribution:

\- Negative candidates: 1,684,429

\- Positive candidates: 185,162



The positive class represents approximately 9.9% of training candidates.



\## 3. Pending Work



\### Model Evaluation

\- \[ ] Validate candidate-miss statistics.

\- \[ ] Execute all five models on actual datasets.

\- \[ ] Collect real GroupKFold Macro F1 results.

\- \[ ] Collect student-test metrics.

\- \[ ] Compare models and select the best classifier.



\### Route Continuity

\- \[ ] Integrate `route\_continuity()` from `src/network.py`.

\- \[ ] Calculate continuity for predicted routes.

\- \[ ] Compare predicted continuity with reference-route continuity.



\### Bayesian Optimisation

\- \[ ] Select the best model based on evaluation results.

\- \[ ] Run Optuna hyperparameter optimisation on real training data.

\- \[ ] Compare tuned and untuned model performance.

\- \[ ] Save final optimisation results.



\### Model Export

\- \[ ] Refit the selected model using all training data.

\- \[ ] Export `models/classifier.joblib`.

\- \[ ] Export `models/classifier\_config.json`.



\### Reporting

\- \[ ] Prepare report Section 3: Model Comparison.

\- \[ ] Prepare report Section 4: Bayesian Hyperparameter Optimisation.

\- \[ ] Include actual experimental metrics and plots.



\## 4. Important Notes



\- Demo data has been used only to test the implementation framework.

\- Demo metrics must not be reported as actual experimental results.

\- Student-test data must not be used for model training or hyperparameter optimisation.

\- Cross-validation must keep trips separated using GroupKFold.

\- Final performance metrics must be generated using the actual datasets.

\- No evaluator dataset has been used.



\## 5. Current Overall Status



The model training and Bayesian optimisation frameworks are implemented.



Actual model training, performance comparison, route continuity integration, final model export and reporting remain pending.



\*\*Overall:\*\* Implementation in progress.



