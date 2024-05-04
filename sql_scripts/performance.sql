--Performance query
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime",
"trading_session", "top_book"
FROM qlike_1W
GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model", "trading_session", "top_book"
UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime",
"trading_session", "top_book"
FROM qlike_1M
GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model", "trading_session", "top_book"
UNION ALL
SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime",
"trading_session", "top_book"
FROM qlike_6M
GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model", "trading_session", "top_book";
.quit


