-- QLIKE per L_train
.load ./math.so
.mode csv
.output ../results/qlike_per_L_train.csv
WITH qlike_1W AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime","trading_session","top_book","L"
	FROM y_1W
),
qlike_1M AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime","trading_session","top_book","L"
	FROM y_1M
),
qlike_6M AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime","trading_session","top_book","L"
	FROM y_6M
),
min_qlike_1W AS (
	SELECT AVG("values") AS "values", "regression","training_scheme","model","vol_regime","trading_session","top_book","L"
	FROM qlike_1W
	GROUP BY "training_scheme", "regression", "model", "trading_session","top_book"
	ORDER BY "values"  LIMIT 1
),
min_qlike_1M AS (
	SELECT AVG("values") AS "values", "regression","training_scheme","model","vol_regime","trading_session","top_book","L"
	FROM qlike_1M
	GROUP BY "training_scheme", "regression", "model", "trading_session","top_book"
	ORDER BY "values"  LIMIT 1
),
min_qlike_6M AS (
	SELECT AVG("values") AS "values", "regression","training_scheme","model","vol_regime","trading_session","top_book","L"
	FROM qlike_6M
	GROUP BY "training_scheme", "regression", "model", "trading_session","top_book"
	ORDER BY "values"  LIMIT 1
),
qlike AS (
	SELECT * FROM min_qlike_1W
	UNION ALL
	SELECT * FROM min_qlike_1M
	UNION ALL
	SELECT * FROM min_qlike_6M
)
SELECT * FROM qlike;
.quit