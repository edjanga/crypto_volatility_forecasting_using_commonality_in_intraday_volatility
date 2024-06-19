.mode csv
.output ../results/y_UAM.csv
SELECT * FROM y_1W WHERE training_scheme="UAM"
UNION ALL
SELECT * FROM y_1M WHERE training_scheme="UAM"
UNION ALL
SELECT * FROM y_6M WHERE training_scheme="UAM";
.output ../results/qlike_UAM.csv
SELECT * FROM qlike_1W WHERE training_scheme="UAM"
UNION ALL
SELECT * FROM qlike_1M WHERE training_scheme="UAM"
UNION ALL
SELECT * FROM qlike_6M WHERE training_scheme="UAM";
