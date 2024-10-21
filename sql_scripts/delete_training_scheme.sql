ATTACH DATABASE '../data_centre/databases/qlike.db' AS qlike;
ATTACH DATABASE '../data_centre/databases/y.db' AS y;
--DELETE FROM y.y_1W WHERE "model"="har_eq" AND "training_scheme"="UAM";
DELETE FROM y.y_1M WHERE "model"="har_eq" AND "training_scheme"="UAM";
--DELETE FROM y.y_6M WHERE "model"="har_eq" AND "training_scheme"="UAM";
.quit
