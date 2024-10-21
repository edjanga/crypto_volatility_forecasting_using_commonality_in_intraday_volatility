-- QLIKE per vol_regime
--1. Low
.mode csv
.output ../results/qlike_per_vol_regime.csv
WITH y_vol_regime_low_1W AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime",
	"trading_session","top_book","L"
	FROM y_1W
	WHERE "vol_regime"="low"
),
y_vol_regime_low_1M AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime",
	"trading_session","top_book","L"
	FROM y_1M
	WHERE "vol_regime"="low"
),
y_vol_regime_low_6M AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime",
	"trading_session","top_book","L"
	FROM y_6M
	WHERE "vol_regime"="low"
),
y_vol_regime_low AS (
	SELECT * FROM y_vol_regime_low_1W
	UNION ALL
	SELECT * FROM y_vol_regime_low_1M
	UNION ALL
	SELECT * FROM y_vol_regime_low_6M
),
qlike_vol_regime_low AS (
	SELECT AVG("values") AS "values","regression","training_scheme","model","vol_regime","trading_session","top_book",
	"L"
	FROM y_vol_regime_low
	GROUP BY "training_scheme", "regression", "model", "trading_session","top_book"
	ORDER BY "values" LIMIT 1
),
--2. Normal
y_vol_regime_normal_1W AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime",
	"trading_session","top_book","L"
	FROM y_1W
	WHERE "vol_regime"="normal"
),
y_vol_regime_normal_1M AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime",
	"trading_session","top_book","L"
	FROM y_1M
	WHERE "vol_regime"="normal"
),
y_vol_regime_normal_6M AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime",
	"trading_session","top_book","L"
	FROM y_6M
	WHERE "vol_regime"="normal"
),
y_vol_regime_normal AS (
	SELECT * FROM y_vol_regime_normal_1W
	UNION ALL
	SELECT * FROM y_vol_regime_normal_1M
	UNION ALL
	SELECT * FROM y_vol_regime_normal_6M
),
qlike_vol_regime_normal AS (
	SELECT AVG("values") AS "values","regression","training_scheme","model","vol_regime","trading_session",
	"top_book","L"
	FROM y_vol_regime_normal
	GROUP BY "training_scheme", "regression", "model", "trading_session","top_book"
	ORDER BY "values" LIMIT 1
),
--3. High
y_vol_regime_high_1W AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime",
	"trading_session","top_book","L"
	FROM y_1W
	WHERE "vol_regime"="high"
),
y_vol_regime_high_1M AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime",
	"trading_session","top_book","L"
	FROM y_1M
	WHERE "vol_regime"="high"
),
y_vol_regime_high_6M AS (
	SELECT "y"/"y_hat" - LN("y"/"y_hat")-1 AS "values","regression","training_scheme","model","symbol","vol_regime",
	"trading_session","top_book","L"
	FROM y_6M
	WHERE "vol_regime"="high"
),
y_vol_regime_high AS (
	SELECT * FROM y_vol_regime_high_1W
	UNION ALL
	SELECT * FROM y_vol_regime_high_1M
	UNION ALL
	SELECT * FROM y_vol_regime_high_6M
),
qlike_vol_regime_high AS (
	SELECT AVG("values") AS "values","regression","training_scheme","model","vol_regime","trading_session","top_book",
	"L"
	FROM y_vol_regime_high
	GROUP BY "training_scheme", "regression", "model", "trading_session","top_book"
	ORDER BY "values" LIMIT 1
),
qlike_vol_regime AS (
	SELECT * FROM qlike_vol_regime_low
	UNION ALL
	SELECT * FROM qlike_vol_regime_normal
	UNION ALL
	SELECT * FROM qlike_vol_regime_high
)
SELECT *  FROM qlike_vol_regime;
.quit