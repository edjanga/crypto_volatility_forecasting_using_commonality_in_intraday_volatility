#!/bin/bash
clear
sqlite3 ../data_centre/databases/qlike.db < ../sql_scripts/delete_duplicated_rows.sql