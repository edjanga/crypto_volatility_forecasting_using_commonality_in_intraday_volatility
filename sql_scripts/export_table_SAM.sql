.mode csv
.output ../results/y_SAM.csv
SELECT * FROM y_1W WHERE training_scheme="SAM"
UNION ALL
SELECT * FROM y_1M WHERE training_scheme="SAM"
UNION ALL
SELECT * FROM y_6M WHERE training_scheme="SAM";
.output ../results/qlike_SAM.csv
SELECT * FROM qlike_1W WHERE training_scheme="SAM"
UNION ALL
SELECT * FROM qlike_1M WHERE training_scheme="SAM"
UNION ALL
SELECT * FROM qlike_6M WHERE training_scheme="SAM";
