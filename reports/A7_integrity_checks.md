# A7 Integrity checks

- PASS: train_unique_zone_hour — dupes=0
- PASS: test_unique_zone_hour — dupes=0
- PASS: zones_are_canonical — train=['Arat Kilo', 'Ayat', 'Bole', 'CMC', 'Gerji', 'Kazanchis', 'Kolfe', 'Lideta', 'Megenagna', 'Merkato', 'Piassa', 'Sarbet']
- PASS: timestamps_in_range_train — 2025-01-01 00:00:00 .. 2025-10-31 23:00:00
- PASS: timestamps_in_range_test — 2025-11-01 00:00:00 .. 2025-11-14 23:00:00
- PASS: no_negative_trips_left — min=0.0
- PASS: weather_columns_complete — all weather features non-null on train
- PASS: same_model_feature_columns — model feature list present on train and test
- PASS: leaky_ops_not_in_model_features — model_features overlap leaky=set()
- PASS: test_row_count_4032 — len=4032