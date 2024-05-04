ATTACH DATABASE '../data_centre/databases/mse.db' AS mse;
ATTACH DATABASE '../data_centre/databases/qlike.db' AS qlike;
ATTACH DATABASE '../data_centre/databases/y.db' AS y;
-- DELETE FROM main.r2_1W WHERE "top_book"=0;
-- DELETE FROM main.r2_1M WHERE "top_book"=0;
-- DELETE FROM main.r2_6M WHERE "top_book"=0;
-- DELETE FROM mse.mse_1W WHERE "training_scheme"="UAM" AND "top_book"=0;
DELETE FROM mse.mse_1M WHERE "training_scheme"="UAM" AND "top_book"=0;
DELETE FROM mse.mse_6M WHERE "training_scheme"="UAM" AND "top_book"=0;
-- DELETE FROM y.y_1W WHERE "top_book"=0;
DELETE FROM y.y_1M WHERE "training_scheme"="UAM" AND "top_book"=0;
DELETE FROM y.y_6M WHERE "training_scheme"="UAM" AND "top_book"=0;
-- DELETE FROM qlike.qlike_1W WHERE "top_book"=0;
DELETE FROM qlike.qlike_1M WHERE "training_scheme"="UAM" AND "top_book"=0;
DELETE FROM qlike.qlike_6M WHERE "training_scheme"="UAM" AND "top_book"=0;
.quit
