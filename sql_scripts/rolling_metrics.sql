/*Rolling metrics query*/
SELECT "timestamp", AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression"
FROM main.r2_1D
GROUP BY "timestamp",  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT "timestamp", AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression"
FROM main.r2_1W
GROUP BY "timestamp",  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT "timestamp", AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression"
FROM main.r2_1M
GROUP BY "timestamp",  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT "timestamp", AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression"
FROM qlike.qlike_1D
GROUP BY "timestamp",  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT "timestamp", AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression"
FROM qlike.qlike_1W
GROUP BY "timestamp",  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT "timestamp", AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression"
FROM qlike.qlike_1M
GROUP BY "timestamp",  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT "timestamp", AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression"
FROM mse.mse_1D
GROUP BY "timestamp",  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT "timestamp", AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression"
FROM mse.mse_1W
GROUP BY "timestamp",  "metric", "training_scheme","L", "regression", "model"
UNION ALL
SELECT "timestamp", AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression"
FROM mse.mse_1M
GROUP BY "timestamp",  "metric", "training_scheme","L", "regression", "model"
ORDER BY "timestamp" ASC;