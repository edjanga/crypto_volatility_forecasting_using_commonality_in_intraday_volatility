.mode csv
.output ../results/y_ClustAM.csv
SELECT * FROM y_1W WHERE training_scheme="ClustAM"
UNION ALL
SELECT * FROM y_1M WHERE training_scheme="ClustAM"
UNION ALL
SELECT * FROM y_6M WHERE training_scheme="ClustAM";
.output ../results/qlike_ClustAM.csv
SELECT * FROM qlike_1W WHERE training_scheme="ClustAM"
UNION ALL
SELECT * FROM qlike_1M WHERE training_scheme="ClustAM"
UNION ALL
SELECT * FROM qlike_6M WHERE training_scheme="ClustAM";
