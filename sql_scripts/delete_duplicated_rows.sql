ATTACH DATABASE '../data_centre/databases/mse.db' AS mse;
ATTACH DATABASE '../data_centre/databases/r2.db' AS r2;
ATTACH DATABASE '../data_centre/databases/qlike.db' AS qlike;
ATTACH DATABASE '../data_centre/databases/y.db' AS y;
-- QLIKE
DELETE FROM qlike.qlike_1W WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model", "values","L","training_scheme","trading_session",
            "transformation","h","metric","regression","vol_regime'","top_book"  ORDER BY ROWID) AS rn
        FROM qlike.qlike_1W
    ) WHERE rn > 1
);
DELETE FROM qlike.qlike_1M WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model", "values","L","training_scheme","trading_session",
            "transformation","h","metric","regression","vol_regime'","top_book"  ORDER BY ROWID) AS rn
        FROM qlike.qlike_1M
    ) WHERE rn > 1
);
DELETE FROM qlike.qlike_6M WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model", "values","L","training_scheme","trading_session",
            "transformation","h","metric","regression","vol_regime'","top_book"  ORDER BY ROWID) AS rn
        FROM qlike.qlike_6M
    ) WHERE rn > 1
);
-- R2
DELETE FROM r2.r2_1W WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model", "values","L","training_scheme","trading_session",
            "transformation","h","metric","regression","vol_regime'","top_book"  ORDER BY ROWID) AS rn
        FROM r2.r2_1W
    ) WHERE rn > 1
);
DELETE FROM r2.r2_1M WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model", "values","L","training_scheme","trading_session",
            "transformation","h","metric","regression","vol_regime'","top_book"  ORDER BY ROWID) AS rn
        FROM r2.r2_1M
    ) WHERE rn > 1
);
DELETE FROM r2.r2_6M WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model", "values","L","training_scheme","trading_session",
            "transformation","h","metric","regression","vol_regime'","top_book"  ORDER BY ROWID) AS rn
        FROM r2.r2_6M
    ) WHERE rn > 1
);
-- MSE
DELETE FROM mse.mse_1W WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model", "values","L","training_scheme","trading_session",
            "transformation","h","metric","regression","vol_regime'","top_book"  ORDER BY ROWID) AS rn
        FROM mse.mse_1W
    ) WHERE rn > 1
);
DELETE FROM mse.mse_1M WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model", "values","L","training_scheme","trading_session",
            "transformation","h","metric","regression","vol_regime'","top_book"  ORDER BY ROWID) AS rn
        FROM mse.mse_1M
    ) WHERE rn > 1
);
DELETE FROM mse.mse_6M WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model", "values","L","training_scheme","trading_session",
            "transformation","h","metric","regression","vol_regime'","top_book"  ORDER BY ROWID) AS rn
        FROM mse.mse_6M
    ) WHERE rn > 1
);
-- y
DELETE FROM y.y_1W WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model","L","training_scheme","trading_session", "transformation",
            "h","regression","vol_regime'","top_book","y","y_hat"  ORDER BY ROWID) AS rn
        FROM y.y_1W
    ) WHERE rn > 1
);
DELETE FROM y.y_1M WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model","L","training_scheme", "trading_session", "transformation",
            "h","regression","vol_regime'","top_book","y","y_hat"  ORDER BY ROWID) AS rn
        FROM y.y_1M
    ) WHERE rn > 1
);
DELETE FROM y.y_6M WHERE ROWID IN (
    SELECT ROWID FROM (
        SELECT ROWID, ROW_NUMBER() OVER (
            PARTITION BY "timestamp", "symbol", "model","L","training_scheme", "trading_session", "transformation",
            "h","regression","vol_regime'","top_book","y","y_hat"  ORDER BY ROWID) AS rn
        FROM y.y_6M
    ) WHERE rn > 1
);
.quit