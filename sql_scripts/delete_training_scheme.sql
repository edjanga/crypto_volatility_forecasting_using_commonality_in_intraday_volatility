ATTACH DATABASE '../data_centre/databases/mse.db' AS mse;
ATTACH DATABASE '../data_centre/databases/qlike.db' AS qlike;
ATTACH DATABASE '../data_centre/databases/y.db' AS y;
--ATTACH DATABASE './data_centre/databases/tstats.db' AS tstats;
--ATTACH DATABASE './data_centre/databases/pvalues.db' AS pvalues;
--ATTACH DATABASE './data_centre/databases/coefficient.db' AS coefficient;
--DELETE FROM main.r2_1W WHERE "training_scheme"="CAM";
--DELETE FROM main.r2_1M WHERE "training_scheme"="CAM";
DELETE FROM main.r2_6M WHERE "training_scheme"="CAM";
--DELETE FROM mse.mse_1W WHERE "training_scheme"="CAM";
--DELETE FROM mse.mse_1M WHERE "training_scheme"="CAM";
DELETE FROM mse.mse_6M WHERE "training_scheme"="CAM";
--DELETE FROM y.y_1W WHERE "training_scheme"="CAM";
--DELETE FROM y.y_1M WHERE "training_scheme"="CAM";
DELETE FROM y.y_6M WHERE "training_scheme"="CAM";
--DELETE FROM qlike.qlike_1W WHERE "training_scheme"="CAM";
--DELETE FROM qlike.qlike_1M WHERE "training_scheme"="CAM";
DELETE FROM qlike.qlike_6M WHERE "training_scheme"="CAM";
--DELETE FROM tstats.tstats_1W WHERE "training_scheme"="CAM";
--DELETE FROM tstats.tstats_1M WHERE "training_scheme"="CAM";
--DELETE FROM tstats.tstats_6M WHERE "training_scheme"="CAM";
--DELETE FROM pvalues.pvalues_1W WHERE "training_scheme"="CAM";
--DELETE FROM pvalues.pvalues_1M WHERE "training_scheme"="CAM";
--DELETE FROM pvalues.pvalues_6M WHERE "training_scheme"="CAM";
--DELETE FROM coefficient.coefficient_1W WHERE "training_scheme"="CAM";
--DELETE FROM coefficient.coefficient_1M WHERE "training_scheme"="CAM";
--DELETE FROM coefficient.coefficient_6M WHERE "training_scheme"="CAM";
.quit
