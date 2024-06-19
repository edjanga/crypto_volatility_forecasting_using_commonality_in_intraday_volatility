.mode csv
.output ../results/y_CAM.csv
SELECT * FROM y_1W WHERE training_scheme="CAM"
UNION ALL
SELECT * FROM y_1M WHERE training_scheme="CAM"
UNION ALL
SELECT * FROM y_6M WHERE training_scheme="CAM";
.output ../results/qlike_CAM.csv
SELECT * FROM qlike_1W WHERE training_scheme="CAM"
UNION ALL
SELECT * FROM qlike_1M WHERE training_scheme="CAM"
UNION ALL
SELECT * FROM qlike_6M WHERE training_scheme="CAM";
