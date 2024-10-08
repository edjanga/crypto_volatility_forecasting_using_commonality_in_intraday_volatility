--Performance query
--SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime",
--"trading_session", "top_book"
--FROM qlike_1W
--GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model", "trading_session", "top_book"
--UNION ALL
--SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime",
--"trading_session", "top_book"
--FROM qlike_1M
--GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model", "trading_session", "top_book"
--UNION ALL
--SELECT AVG("values") AS "values", "metric", "model", "L", "training_scheme", "regression", "vol_regime",
--"trading_session", "top_book"
--FROM qlike_6M
--GROUP BY  "metric", "training_scheme","L", "regression", "vol_regime", "model", "trading_session", "top_book";*/
.mode csv
.output ../results/qlike.csv
WITH Aggregated_Y1W AS (
    SELECT
        strftime("%Y-%m-%d %H:", "timestamp") ||
        CASE WHEN cast(strftime("%M", "timestamp") as integer) < 30 THEN "00" ELSE "30" END as timestamp,
        SUM("y") AS "y", SUM("y_hat") AS "y_hat",
        "symbol","training_scheme", "regression","model","trading_session","top_book", "L", "h"
    FROM y_1W
    GROUP BY "symbol", "training_scheme", "regression",  "model", "trading_session","top_book", "timestamp"
),
Aggregated_Y1M AS (
    SELECT
        strftime("%Y-%m-%d %H:", "timestamp") ||
        CASE WHEN cast(strftime("%M", "timestamp") as integer) < 30 THEN "00" ELSE "30" END as timestamp,
        SUM("y") AS "y", SUM("y_hat") AS "y_hat",
        "symbol","training_scheme", "regression","model","trading_session","top_book", "L", "h"
    FROM y_1M
    GROUP BY "symbol", "training_scheme", "regression",  "model", "trading_session","top_book", "timestamp"
),
Aggregated_Y6M AS (
    SELECT
        strftime("%Y-%m-%d %H:", "timestamp") ||
        CASE WHEN cast(strftime("%M", "timestamp") as integer) < 30 THEN "00" ELSE "30" END as timestamp,
        SUM("y") AS "y", SUM("y_hat") AS "y_hat",
        "symbol","training_scheme", "regression","model","trading_session","top_book", "L", "h"
    FROM y_6M
    GROUP BY "symbol", "training_scheme", "regression",  "model", "trading_session","top_book", "timestamp"
),
Computed_Values AS (
    SELECT AVG(y/y_hat) - (y/y_hat) - 1 AS "values",
           "model", "training_scheme", "regression", "trading_session", "top_book", "L", "h"
    FROM Aggregated_Y1W
    UNION ALL
    SELECT AVG(y/y_hat) - (y/y_hat) - 1 AS "values",
           "model", "training_scheme", "regression", "trading_session", "top_book", "L", "h"
    FROM Aggregated_Y1M
    UNION ALL
    SELECT AVG(y/y_hat) - (y/y_hat) - 1 AS "values",
           "model", "training_scheme", "regression", "trading_session", "top_book", "L", "h"
    FROM Aggregated_Y6M
)
SELECT
    MIN("values") AS "values", "model", "regression","training_scheme",
    "trading_session","top_book", "L", "h"
FROM (
    SELECT
        AVG("values") AS "values", "model", "regression", "training_scheme",
        "trading_session", "top_book", "L", "h"
    FROM Computed_Values
    GROUP BY "training_scheme", "regression", "model", "trading_session", "top_book"
);
.quit


