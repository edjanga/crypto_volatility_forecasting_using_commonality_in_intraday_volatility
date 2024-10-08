ATTACH DATABASE '../data_centre/databases/qlike.db' AS qlike;
ATTACH DATABASE '../data_centre/databases/y.db' AS y;
DELETE FROM y.y_1W WHERE "model"="har_eq";
DELETE FROM y.y_1M WHERE "model"="har_eq";
DELETE FROM y.y_6M WHERE "model"="har_eq";
DELETE FROM qlike.qlike_1W WHERE "model"="har_eq";
DELETE FROM qlike.qlike_1M WHERE "model"="har_eq";
DELETE FROM qlike.qlike_6M WHERE "model"="har_eq";
.quit
