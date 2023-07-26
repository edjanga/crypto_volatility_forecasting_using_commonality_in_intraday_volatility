/*Performance query*/
ATTACH DATABASE './data_centre/databases/mse.db' AS mse;
ATTACH DATABASE './data_centre/databases/qlike.db' AS qlike;
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression" FROM main.r2_1D
GROUP BY  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression" FROM main.r2_1W
GROUP BY  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression" FROM main.r2_1M
GROUP BY  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression" FROM qlike.qlike_1D
GROUP BY  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression" FROM qlike.qlike_1W
GROUP BY  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression" FROM qlike.qlike_1M
GROUP BY  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT SUM("values") AS "values", "metric", "model", "L", "training_scheme", "regression" FROM mse.mse_1D
GROUP BY  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT SUM("values") AS "values", "metric", "model", "L", "training_scheme", "regression" FROM mse.mse_1W
GROUP BY  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT SUM("values") AS "values", "metric", "model", "L", "training_scheme", "regression" FROM mse.mse_1M
GROUP BY  "metric", "training_scheme","L", "regression", "model";
.quit


