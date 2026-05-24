# ML HyperLab - Regression Support Worklog

---
Task ID: regression-support
Agent: main
Task: Add full regression support to ML HyperLab (was classification-only)

Work Log:
- Verified ml-algorithms.ts already had regression algorithms, metrics, datasets, and training functions implemented
- Added REGRESSION_ALGORITHMS (6 algos with Spanish descriptions) and REGRESSION_DATASET_PRESETS (2 datasets) to ml-config.ts
- Updated csv-importer.ts with detectTaskType() function and taskType field in ImportedDataset interface
- Updated page.tsx with task type toggle (Clasificación/Regresión), regression-specific state, handlers, metrics cards (MSE, RMSE, MAE, R²), and visualizations (scatter plot + residual plot)
- Added RegressionScatterCanvas component (actual vs predicted values)
- Added ResidualPlotCanvas component (residuals vs predicted values)
- Updated MetricCard to support both percentage and raw value display modes
- Added auto-detection of regression vs classification when importing CSV files
- Fixed pre-existing TypeScript bugs in ml-algorithms.ts (yMin → yMin2) and csv-importer.ts (preview variable scoping)
- Build successful with no TypeScript errors

Stage Summary:
- ML HyperLab now supports BOTH classification and regression
- Toggle switch in the Lab tab switches between modes
- Regression mode has: 6 regressors, 2 regression datasets, MSE/RMSE/MAE/R² metrics, scatter plot + residual plot visualizations
- CSV import auto-detects task type based on target column characteristics
- All UI text is in Spanish
