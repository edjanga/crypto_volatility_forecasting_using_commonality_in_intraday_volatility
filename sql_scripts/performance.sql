/*Performance query*/
ATTACH DATABASE '../data_centre/databases/mse.db' AS mse;
ATTACH DATABASE '../data_centre/databases/qlike.db' AS qlike;
--SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime"
--FROM main.r2_1W
--GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model"
--UNION ALL
--SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime"
--FROM main.r2_1M
--GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model"
--UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime"
FROM main.r2_6M
GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model"
UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime"
FROM qlike.qlike_1W
GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model"
UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime"
FROM qlike.qlike_1M
GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model"
UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime"
FROM qlike.qlike_6M
GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model"
UNION ALL
SELECT SUM("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime"
FROM mse.mse_1W
GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model"
UNION ALL
SELECT SUM("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime"
FROM mse.mse_1M
GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model"
UNION ALL
SELECT SUM("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime"
FROM mse.mse_6M
GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model";
.quit


