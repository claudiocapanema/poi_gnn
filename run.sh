#!/bin/bash
echo "Argument: "$1

# common variables
USERS_STEPS_BASE_DIR="/media/claudio/Data/backup_linux/Documentos/users_steps_datasets/"
USERS_STEPS_OUTPUT_BASE_DIR="/media/claudio/Data/backup_win_hd/Downloads/doutorado/users_steps_output/"
USERS_194_STEPS_FILENAME="${USERS_STEPS_BASE_DIR}194_users_data.csv"
USERS_10_MIL_MAX_500_POINTS="${USERS_STEPS_BASE_DIR}df_mais_de_10_mil_limite_500_pontos.csv"
DETECTED_USERS_10_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}detected_df_mais_de_10_mil_limite_500_pontos.csv"
CLASSIFIED_USERS_10_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}classified_df_mais_de_10_mil_limite_500_pontos.csv"
USERS_5_MIL_MAX_500_POINTS="${USERS_STEPS_BASE_DIR}df_mais_de_5_mil_limite_500_pontos.csv"
DETECTED_USERS_5_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}detected_df_mais_de_5_mil_limite_500_pontos.csv"
CLASSIFIED_USERS_5_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}classified_df_mais_de_5_mil_limite_500_pontos.csv"
BASE_DIR="/home/claudio/Documentos/users_steps_datasets/tipo_0_2/user_steps_tipo_0_2/"
USERS_STEPS_FILENAME="/home/guilherme/Documentos/Users_filtrados_por_regiao/frequent_users.csv"
POI_DETECTION_FILENAME="/home/guilherme/Documentos/users_detected_pois.csv"
POI_CLASSIFICATION_FILENAME="/home/guilherme/Documentos/users_classified_pois.csv"
GROUND_TRUTH="/home/guilherme/Documentos/Users_filtrados_por_regiao/Users_rotulated/rotulated_manual_edit.csv"
SHAPEFILE_FILENAME="${BASE_DIR}/BR_shapefile/BRMUE250GC_SIR.shp"
USERS_CHECKIN_FILENAME="${BASE_DIR}/weeplaces/weeplace_1000.csv"
USERS_CHECKIN_FILENAME_BASELINE_HMRM="/home/guilherme/Documentos/Analisar_dataset_weeplaces_7_8_meses/users_7_8_months_clean.csv"
FEATURES_FILENAME_HMRM="/home/guilherme/Documentos/new_users_steps_dataset/features_hmrm_500_users.csv"
ADJACENCY_MATRIX_BASE_FILENAME="adjacency_matrix"
FEATURES_MATRIX_BASE_FILENAME="features_matrix"
DISTANCE_MATRIX_BASE_FILENAME="distance_matrix"
DURATION_MATRIX_BASE_FILENAME="duration_matrix"
SEQUENCE_MATRIX_BASE_FILENAME="sequence_matrix"

# Dense foursquare
DENSE_FOURSQUARE_BASE_DIR="/media/claudio/Data/backup_win_hd/Downloads/doutorado/fourquare_thiago/"
DENSE_FOURSQUARE_LOCAL_DATETIME="${DENSE_FOURSQUARE_BASE_DIR}dense_foursquare_local_datetime_us_brazil_states_9_categories.csv"
DENSE_FOURSQUARE_ADJACENCY_FILENAME="adjacency_matrix_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_ADJACENCY_WEEK_FILENAME="adjacency_matrix_weekday_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_ADJACENCY_WEEKEND_FILENAME="adjacency_matrix_weekend_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_TEMPORAL_FILENAME="features_matrix_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_TEMPORAL_WEEK_FILENAME="features_matrix_weekday_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_TEMPORAL_WEEKEND_FILENAME="features_matrix_weekend_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_DISTANCE_FILENAME="distance_matrix_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_DISTANCE_WEEK_FILENAME="distance_matrix_weekday_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_DISTANCE_WEEKEND_FILENAME="distance_matrix_weekend_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_DURATION_FILENAME="duration_matrix_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_DURATION_WEEK_FILENAME="duration_matrix_weekday_not_directed_48_9_categories_BR.csv"
DENSE_FOURSQUARE_DURATION_WEEKEND_FILENAME="duration_matrix_weekend_not_directed_48_9_categories_BR.csv"

# Global foursquare
GLOBAL_FOURSQUARE_BASE_DIR="/media/claudio/Data/backup_win_hd/Downloads/doutorado/global_foursquare/dataset_TIST2015/"
GLOBAL_FOURSQUARE_8_CATEGORIES_DATASET="${GLOBAL_FOURSQUARE_BASE_DIR}dataset_TIST2015_Checkins_with_Pois_8_categories_local_datetime_months_4_and_5_year_2012.csv"
GLOBAL_FOURSQUARE_8_CATEGORIES_BR_US_JP_2012_DATASET="${GLOBAL_FOURSQUARE_BASE_DIR}dataset_TIST2015_Checkins_with_Pois_8_categories_local_datetime_br_us_jp_2012.csv"
GLOBAL_FOURSQUARE_8_CATEGORIES_BR_US_JP_2012_2013_DATASET="${GLOBAL_FOURSQUARE_BASE_DIR}dataset_TIST2015_Checkins_with_Pois_8_categories_local_datetime_br_us_jp_2012_2013.csv"
GLOBAL_FOURSQUARE_8_CATEGORIES_BR_US_JP_2012_2013_DATASET_WITH_STATE_AND_CITIES="${GLOBAL_FOURSQUARE_BASE_DIR}dataset_TIST2015_Checkins_with_Pois_8_categories_event_excluded_local_datetime_br_us_jp_2012_2013_with_state_and_cities.csv"
CHECK_INS_LOCAL_DATETIME_9_CATEGORIES_EVENT_EXCLUDED_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_US_JP_WITH_STATE_AND_CITY="${GLOBAL_FOURSQUARE_BASE_DIR}dataset_TIST2015_Checkins_with_Pois_9_categories_event_excluded_local_datetime_br_us_jp_2012_2013_with_state_and_cities.csv"
GLOBAL_FOURSQUARE_DIRECTED_FOLDER="matrizes/directed/"
GLOBAL_FOURSQUARE_NOT_DIRECTED_FOLDER="matrizes/not_directed/"

## br

GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_ADJACENCY_MATRIX_FILENAME="adjacency_matrix_not_directed_48_8_categories_from_2012_month_4_to_2013_month_9_BR.csv"
GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_FEATURE_MATRIX_FILENAME="features_matrix_not_directed_48_8_categories_from_2012_month_4_to_2013_month_9_BR.csv"
GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_SEQUENCES_MATRIX_FILENAME="sequence_matrix_not_directed_48_8_categories_from_2012_month_4_to_2013_month_9_BR.csv"
# week
GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_ADJACENCY_WEEK_MATRIX_FILENAME="adjacency_matrix_weekday_not_directed_48_8_categories_from_2012_month_4_to_2013_month_9_BR.csv"
GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_FEATURE_WEEK_MATRIX_FILENAME="features_matrix_weekday_not_directed_48_8_categories_from_2012_month_4_to_2013_month_9_BR.csv"
GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_SEQUENCES_WEEK_MATRIX_FILENAME="sequence_matrix_weekday_not_directed_48_8_categories_from_2012_month_4_to_2013_month_9_BR.csv"
# weekend
GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_ADJACENCY_WEEKEND_MATRIX_FILENAME="adjacency_matrix_weekend_not_directed_48_8_categories_from_2012_month_4_to_2013_month_9_BR.csv"
GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_FEATURE_WEEKEND_MATRIX_FILENAME="features_matrix_weekend_not_directed_48_8_categories_from_2012_month_4_to_2013_month_9_BR.csv"
GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_SEQUENCES_WEEKEND_MATRIX_FILENAME="sequence_matrix_weekend_not_directed_48_8_categories_from_2012_month_4_to_2013_month_9_BR.csv"
# directed
GLOBAL_FOURSQUARE_DIRECTED_8_CATEGORIES_MONTHS_4_12_BR_ADJACENCY_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}matrizes/directed/adjacency_matrix_directed_48_8_categories_months_4_12_BR.csv"
GLOBAL_FOURSQUARE_DIRECTED_8_CATEGORIES_MONTHS_4_12_BR_FEATURE_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}matrizes/directed/features_matrix_directed_48_8_categories_months_4_12_BR.csv"
GLOBAL_FOURSQUARE_DIRECTED_8_CATEGORIES_MONTHS_4_12_BR_SEQUENCES_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}matrizes/directed/sequence_matrix_directed_48_8_categories_months_4_12_BR.csv"

# gpr
GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_BR_ADJACENCY_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/adjacency_matrix_directed_48_8_categories_BR.csv"
GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_BR_FEATURE_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/features_matrix_directed_48_8_categories_BR.csv"
GLOBAL_FOURSQUARE_GPR_8_CATEGORIES_BR_SEQUENCES_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/user_poi_vector_directed_48_8_categories_BR.csv"
GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_MONTHS_4_12_2012_BR_ADJACENCY_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/adjacency_matrix_directed_48_8_categories_months_4_12_BR.csv"
GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_MONTHS_4_12_2012_BR_FEATURE_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/features_matrix_directed_48_8_categories_months_4_12_BR.csv"
GLOBAL_FOURSQUARE_GPR_8_CATEGORIES_MONTHS_4_12_2012_BR_SEQUENCES_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/user_poi_vector_directed_48_8_categories_months_4_12_BR.csv"

GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_MONTHS_4_12_2012_US_ADJACENCY_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/adjacency_matrix_directed_48_8_categories_months_4_12_US.csv"
GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_MONTHS_4_12_2012_US_FEATURE_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/features_matrix_directed_48_8_categories_months_4_12_US.csv"
GLOBAL_FOURSQUARE_GPR_8_CATEGORIES_MONTHS_4_12_2012_US_SEQUENCES_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/user_poi_vector_directed_48_8_categories_months_4_12_US.csv"

GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_US_ADJACENCY_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/adjacency_matrix_directed_48_8_categories_US.csv"
GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_JP_ADJACENCY_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/adjacency_matrix_directed_48_8_categories_JP.csv"
GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_JP_FEATURE_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/features_matrix_directed_48_8_categories_JP.csv"
GLOBAL_FOURSQUARE_GPR_8_CATEGORIES_JP_SEQUENCES_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/user_poi_vector_directed_48_8_categories_JP.csv"
GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_US_FEATURE_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/features_matrix_directed_48_8_categories_US.csv"
GLOBAL_FOURSQUARE_GPR_8_CATEGORIES_US_SEQUENCES_MATRIX_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}gpr/directed/user_poi_vector_directed_48_8_categories_US.csv"

GLOBAL_FOURQUARE_USER_METRICS_FILENAME_BR="files/global_foursquare_user_metrics_BR.csv"
GLOBAL_FOURQUARE_USER_METRICS_FILENAME_2012_2013_BR="files/global_foursquare_user_metrics_BR_2012_2013.csv"
GLOBAL_FOURQUARE_USER_METRICS_FILENAME_US="files/global_foursquare_user_metrics_US.csv"

# gowalla
GOWALLA_BASE_DIR="/media/claudio/Data/backup_win_hd/Downloads/doutorado/gowalla/matrices_poi_categorization/"
GOWALLA_LOCAL_DATETIME_US="/media/claudio/Data/backup_win_hd/Downloads/doutorado/gowalla/gowalla_checkins_7_categories_local_datetime_columns_reduced_us.csv"
GOWALLA_DIRECTED_FOLDER=${GOWALLA_BASE_DIR}"matrices/directed/"
GOWALLA_NOT_DIRECTED_FOLDER=${GOWALLA_BASE_DIR}"matrices/not_directed/"
GOWALLA_ADJACENCY_FILENAME_BR="adjacency_matrix_not_directed_48_9_categories_BR.csv"
GOWALLA_ADJACENCY_WEEK_FILENAME_BR="adjacency_matrix_weekday_not_directed_48_9_categories_BR.csv"
GOWALLA_ADJACENCY_WEEKEND_FILENAME_BR="adjacency_matrix_weekend_not_directed_48_9_categories_BR.csv"
GOWALLA_TEMPORAL_FILENAME_BR="features_matrix_not_directed_48_9_categories_BR.csv"
GOWALLA_TEMPORAL_WEEK_FILENAME_BR="features_matrix_weekday_not_directed_48_9_categories_BR.csv"
GOWALLA_TEMPORAL_WEEKEND_FILENAME_BR="features_matrix_weekend_not_directed_48_9_categories_BR.csv"
GOWALLA_DISTANCE_FILENAME_BR="distance_matrix_not_directed_48_9_categories_BR.csv"
GOWALLA_DISTANCE_WEEK_FILENAME_BR="distance_matrix_weekday_not_directed_48_9_categories_BR.csv"
GOWALLA_DISTANCE_WEEKEND_FILENAME_BR="distance_matrix_weekend_not_directed_48_9_categories_BR.csv"
GOWALLA_DURATION_FILENAME_BR="duration_matrix_not_directed_48_9_categories_BR.csv"
GOWALLA_DURATION_WEEK_FILENAME_BR="duration_matrix_weekday_not_directed_48_9_categories_BR.csv"
GOWALLA_DURATION_WEEKEND_FILENAME_BR="duration_matrix_weekend_not_directed_48_9_categories_BR.csv"

# us
GOWALLA_ADJACENCY_FILENAME_US="adjacency_matrix_not_directed_48_7_categories_US.csv"
GOWALLA_ADJACENCY_WEEK_FILENAME_US="adjacency_matrix_weekday_not_directed_48_7_categories_US.csv"
GOWALLA_ADJACENCY_WEEKEND_FILENAME_US="adjacency_matrix_weekend_not_directed_48_7_categories_US.csv"
GOWALLA_TEMPORAL_FILENAME_US="features_matrix_not_directed_48_7_categories_US.csv"
GOWALLA_TEMPORAL_WEEK_FILENAME_US="features_matrix_weekday_not_directed_48_7_categories_US.csv"
GOWALLA_TEMPORAL_WEEKEND_FILENAME_US="features_matrix_weekend_not_directed_48_7_categories_US.csv"
GOWALLA_DISTANCE_FILENAME_US="distance_matrix_not_directed_48_7_categories_US.csv"
GOWALLA_DISTANCE_WEEK_FILENAME_US="distance_matrix_weekday_not_directed_48_7_categories_US.csv"
GOWALLA_DISTANCE_WEEKEND_FILENAME_US="distance_matrix_weekend_not_directed_48_7_categories_US.csv"
GOWALLA_DURATION_FILENAME_US="duration_matrix_not_directed_48_7_categories_US.csv"
GOWALLA_DURATION_WEEK_FILENAME_US="duration_matrix_weekday_not_directed_48_7_categories_US.csv"
GOWALLA_DURATION_WEEKEND_FILENAME_US="duration_matrix_weekend_not_directed_48_7_categories_US.csv"

# poi transactions analysis
# antigo
POI_TRANSACTIONS_ANALYSIS_OLD_8_CATEGORIES_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}dataset_TIST2015_Checkins_with_Pois_8_categories_local_datetime_br_us_ca_ny_jp_2012_2013_with_state_and_cities.csv"
POI_TRANSACTIONS_ANALYSIS_8_CATEGORIES_EVENT_EXCLUDED_FILENAME="${GLOBAL_FOURSQUARE_BASE_DIR}dataset_TIST2015_Checkins_with_Pois_8_categories_event_excluded_local_datetime_br_us_jp_2012_2013_with_state_and_cities.csv"
TRANSACTIONS_BASE_DIR="${GLOBAL_FOURSQUARE_BASE_DIR}transactions/"

# categories_type = 'osm' or 'reduced_osm'
#MATRIX_GENERATION_FOR_POI_CATEGORIZATION_CONFIG='{
#          "job": "matrix_generation_for_poi_categorization",
#          "users_checkin_filename": "'$WEEPLACES_DATASET'",
#          "base_dir":"'$WEEPLACES_BASE_DIR'",
#          "directed_folder":"'$WEEPLACES_DIRECTED_FOLDER'",
#          "not_directed_folder":"'$WEEPLACES_NOT_DIRECTED_FOLDER'",
#          "adjacency_matrix_base_filename":"'$ADJACENCY_MATRIX_BASE_FILENAME'",
#          "features_matrix_base_filename":"'$FEATURES_MATRIX_BASE_FILENAME'",
#          "directed":"no",
#          "dataset_name":"weeplaces",
#          "categories_type":"7_categories_osm"
#          }'

#MATRIX_GENERATION_FOR_POI_CATEGORIZATION_CONFIG='{
#          "job": "matrix_generation_for_poi_categorization",
#          "users_checkin_filename": "'$WEEPLACES_DATASET'",
#          "base_dir":"'$WEEPLACES_BASE_DIR'",
#          "directed_folder":"'$WEEPLACES_DIRECTED_FOLDER'",
#          "not_directed_folder":"'$WEEPLACES_NOT_DIRECTED_FOLDER'",
#          "adjacency_matrix_base_filename":"'$ADJACENCY_MATRIX_BASE_FILENAME'",
#          "features_matrix_base_filename":"'$FEATURES_MATRIX_BASE_FILENAME'",
#          "hour48":"yes",
#          "personal_features_matrix":"no",
#          "sequence_matrix_base_filename":"'$SEQUENCE_MATRIX_BASE_FILENAME'",
#          "directed":"yes",
#          "dataset_name":"weeplaces",
#          "pattern_matrices":"no",
#          "categories_type":"7_categories_osm"
#          }'

MATRIX_GENERATION_FOR_POI_CATEGORIZATION_CONFIG='{
          "job": "matrix_generation_for_poi_categorization",
          "users_checkin_filename": "'$GOWALLA_LOCAL_DATETIME_US'",
          "base_dir":"'$GOWALLA_BASE_DIR'",
          "directed_folder":"directed/",
          "not_directed_folder":"not_directed/",
          "adjacency_matrix_base_filename":"'$ADJACENCY_MATRIX_BASE_FILENAME'",
          "features_matrix_base_filename":"'$FEATURES_MATRIX_BASE_FILENAME'",
          "distance_matrix_base_filename":"'$DISTANCE_MATRIX_BASE_FILENAME'",
          "duration_matrix_base_filename":"'$DURATION_MATRIX_BASE_FILENAME'",
          "hour48":"yes",
          "personal_features_matrix":"no",
          "sequence_matrix_base_filename":"'$SEQUENCE_MATRIX_BASE_FILENAME'",
          "country":"United States",
          "state":"TEXAS",
          "different_venues":"yes",
          "max_time_between_records": "",
          "top_users":"40000",
          "directed":"no",
          "dataset_name":"gowalla",
          "pattern_matrices":"yes",
          "categories_type":"7_categories"
          }'

#MATRIX_GENERATION_FOR_POI_CATEGORIZATION_CONFIG='{
#          "job": "matrix_generation_for_poi_categorization",
#          "users_checkin_filename": "'$DENSE_FOURSQUARE_LOCAL_DATETIME'",
#          "base_dir":"'$DENSE_FOURSQUARE_BASE_DIR'",
#          "directed_folder":"'$GLOBAL_FOURSQUARE_DIRECTED_FOLDER'",
#          "not_directed_folder":"'$GLOBAL_FOURSQUARE_NOT_DIRECTED_FOLDER'",
#          "adjacency_matrix_base_filename":"'$ADJACENCY_MATRIX_BASE_FILENAME'",
#          "features_matrix_base_filename":"'$FEATURES_MATRIX_BASE_FILENAME'",
#          "distance_matrix_base_filename":"'$DISTANCE_MATRIX_BASE_FILENAME'",
#          "duration_matrix_base_filename":"'$DURATION_MATRIX_BASE_FILENAME'",
#          "hour48":"yes",
#          "personal_features_matrix":"no",
#          "sequence_matrix_base_filename":"'$SEQUENCE_MATRIX_BASE_FILENAME'",
#          "country":"United States",
#          "state":"CALIFORNIA",
#          "different_venues":"yes",
#          "max_time_between_records": "",
#          "top_users":"40000",
#          "directed":"no",
#          "dataset_name":"dense_foursquare",
#          "pattern_matrices":"yes",
#          "categories_type":"9_categories"
#          }'

#CATEGORIZATION_CONFIG='{
#          "job": "categorization",
#          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
#          "users_steps_filename": "'$USERS_STEPS_FILENAME'",
#          "ground_truth": "'$GROUND_TRUTH'",
#          "users_checkin_filename": "'$GLOBAL_FOURSQUARE_8_CATEGORIES_DATASET'",
#          "base_dir":"'$GLOBAL_FOURSQUARE_BASE_DIR'different_venues/matrizes/",
#          "adjacency_matrix_filename": "'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_ADJACENCY_MATRIX_FILENAME'",
#          "feature_matrix_filename": "'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_FEATURE_MATRIX_FILENAME'",
#          "path_matrix_filename":"'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_SEQUENCES_MATRIX_FILENAME'",
#          "adjacency_matrix_week_filename": "'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_ADJACENCY_WEEK_MATRIX_FILENAME'",
#          "feature_matrix_week_filename": "'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_FEATURE_WEEK_MATRIX_FILENAME'",
#          "path_matrix_week_filename":"'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_SEQUENCES_WEEK_MATRIX_FILENAME'",
#          "adjacency_matrix_weekend_filename": "'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_ADJACENCY_WEEKEND_MATRIX_FILENAME'",
#          "feature_matrix_weekend_filename": "'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_FEATURE_WEEKEND_MATRIX_FILENAME'",
#          "path_matrix_weekend_filename":"'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_SEQUENCES_WEEKEND_MATRIX_FILENAME'",
#          "user_metrics":"'$GLOBAL_FOURQUARE_USER_METRICS_FILENAME_2012_2013_BR'",
#          "graph_type":"not_directed",
#          "dataset_name":"global_foursquare",
#          "max_time_between_records":"",
#          "country":"BR",
#          "state":"",
#          "version":"PATH",
#          "categories_type":"8_categories"
#          }'

#CATEGORIZATION_CONFIG='{
#          "job": "categorization",
#          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
#          "users_steps_filename": "'$USERS_STEPS_FILENAME'",
#          "ground_truth": "'$GROUND_TRUTH'",
#          "base_dir":"'$DENSE_FOURSQUARE_BASE_DIR'different_venues/matrizes/",
#          "adjacency_matrix_filename": "'$DENSE_FOURSQUARE_ADJACENCY_FILENAME'",
#          "adjacency_matrix_week_filename": "'$DENSE_FOURSQUARE_ADJACENCY_WEEK_FILENAME'",
#          "adjacency_matrix_weekend_filename": "'$DENSE_FOURSQUARE_ADJACENCY_WEEKEND_FILENAME'",
#          "temporal_matrix_filename": "'$DENSE_FOURSQUARE_TEMPORAL_FILENAME'",
#          "temporal_matrix_week_filename": "'$DENSE_FOURSQUARE_TEMPORAL_WEEK_FILENAME'",
#          "temporal_matrix_weekend_filename": "'$DENSE_FOURSQUARE_TEMPORAL_WEEKEND_FILENAME'",
#          "distance_matrix_filename":"'$DENSE_FOURSQUARE_DISTANCE_FILENAME'",
#          "distance_matrix_week_filename":"'$DENSE_FOURSQUARE_DISTANCE_WEEK_FILENAME'",
#          "distance_matrix_weekend_filename":"'$DENSE_FOURSQUARE_DISTANCE_WEEKEND_FILENAME'",
#          "duration_matrix_filename":"'$DENSE_FOURSQUARE_DURATION_FILENAME'",
#          "duration_matrix_week_filename":"'$DENSE_FOURSQUARE_DURATION_WEEK_FILENAME'",
#          "duration_matrix_weekend_filename":"'$DENSE_FOURSQUARE_DURATION_WEEKEND_FILENAME'",
#          "graph_type":"not_directed",
#          "dataset_name":"dense_foursquare",
#          "country":"United States",
#          "state":"CALIFORNIA",
#          "version":"normal",
#          "categories_type":"9_categories"
#          }'

#CATEGORIZATION_CONFIG='{
#          "job": "categorization",
#          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
#          "users_steps_filename": "'$USERS_STEPS_FILENAME'",
#          "ground_truth": "'$GROUND_TRUTH'",
#          "base_dir":"'$GOWALLA_BASE_DIR'different_venues/matrizes/",
#          "adjacency_matrix_filename": "'$DENSE_FOURSQUARE_ADJACENCY_FILENAME'",
#          "adjacency_matrix_week_filename": "'$DENSE_FOURSQUARE_ADJACENCY_WEEK_FILENAME'",
#          "adjacency_matrix_weekend_filename": "'$DENSE_FOURSQUARE_ADJACENCY_WEEKEND_FILENAME'",
#          "temporal_matrix_filename": "'$DENSE_FOURSQUARE_TEMPORAL_FILENAME'",
#          "temporal_matrix_week_filename": "'$DENSE_FOURSQUARE_TEMPORAL_WEEK_FILENAME'",
#          "temporal_matrix_weekend_filename": "'$DENSE_FOURSQUARE_TEMPORAL_WEEKEND_FILENAME'",
#          "distance_matrix_filename":"'$DENSE_FOURSQUARE_DISTANCE_FILENAME'",
#          "distance_matrix_week_filename":"'$DENSE_FOURSQUARE_DISTANCE_WEEK_FILENAME'",
#          "distance_matrix_weekend_filename":"'$DENSE_FOURSQUARE_DISTANCE_WEEKEND_FILENAME'",
#          "duration_matrix_filename":"'$DENSE_FOURSQUARE_DURATION_FILENAME'",
#          "duration_matrix_week_filename":"'$DENSE_FOURSQUARE_DURATION_WEEK_FILENAME'",
#          "duration_matrix_weekend_filename":"'$DENSE_FOURSQUARE_DURATION_WEEKEND_FILENAME'",
#          "graph_type":"not_directed",
#          "dataset_name":"dense_foursquare",
#          "country":"United States",
#          "state":"CALIFORNIA",
#          "version":"normal",
#          "categories_type":"9_categories"
#          }'

CATEGORIZATION_CONFIG='{
          "job": "categorization",
          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
          "users_steps_filename": "'$USERS_STEPS_FILENAME'",
          "ground_truth": "'$GROUND_TRUTH'",
          "base_dir":"'$GOWALLA_BASE_DIR'different_venues/",
          "adjacency_matrix_filename": "'$GOWALLA_ADJACENCY_FILENAME_US'",
          "temporal_matrix_filename": "'$GOWALLA_TEMPORAL_FILENAME_US'",
          "distance_matrix_filename":"'$GOWALLA_DISTANCE_FILENAME_US'",
          "duration_matrix_filename":"'$GOWALLA_DURATION_FILENAME_US'",
          "adjacency_matrix_week_filename": "'$GOWALLA_ADJACENCY_WEEK_FILENAME_US'",
          "adjacency_matrix_weekend_filename": "'$GOWALLA_ADJACENCY_WEEKEND_FILENAME_US'",
          "temporal_matrix_week_filename": "'$GOWALLA_TEMPORAL_WEEK_FILENAME_US'",
          "temporal_matrix_weekend_filename": "'$GOWALLA_TEMPORAL_WEEKEND_FILENAME_US'",
          "distance_matrix_week_filename":"'$GOWALLA_DISTANCE_WEEK_FILENAME_US'",
          "distance_matrix_weekend_filename":"'$GOWALLA_DISTANCE_WEEKEND_FILENAME_US'",
          "duration_matrix_week_filename":"'$GOWALLA_DURATION_WEEK_FILENAME_US'",
          "duration_matrix_weekend_filename":"'$GOWALLA_DURATION_WEEKEND_FILENAME_US'",
          "graph_type":"not_directed",
          "dataset_name":"gowalla",
          "country":"US",
          "state":"TEXAS",
          "version":"normal",
          "categories_type":"7_categories"
          }'


BASELINES_CONFIG='{
          "job": "hmrm_baseline",
          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
          "users_steps_filename": "'$USERS_STEPS_FILENAME_BASELINE_HMRM'",
          "ground_truth": "'$GROUND_TRUTH'",
          "users_checkin_filename": "'$USERS_CHECKIN_FILENAME_BASELINE_HMRM'",
          "features_filename": "'$FEATURES_FILENAME_HMRM'",
          "weeplaces?": "Yes"
          }'

#CATEGORIZATION_BASELINES_CONFIG='{
#          "job": "categorization_baselines",
#          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
#          "users_steps_filename": "'$USERS_STEPS_FILENAME'",
#          "ground_truth": "'$GROUND_TRUTH'",
#          "base_dir":"'$GLOBAL_FOURSQUARE_BASE_DIR'matrizes/",
#          "adjacency_matrix_filename": "'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_ADJACENCY_MATRIX_FILENAME'",
#          "feature_matrix_filename": "'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_FEATURE_MATRIX_FILENAME'",
#          "sequence_matrix":"'$GLOBAL_FOURSQUARE_NOT_DIRECTED_8_CATEGORIES_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_SEQUENCES_MATRIX_FILENAME'",
#          "user_metrics":"'$GLOBAL_FOURQUARE_USER_METRICS_FILENAME_2012_2013_BR'",
#          "graph_type":"not_directed",
#          "dataset_name":"global_foursquare",
#          "country":"BR",
#          "categories_type":"8_categories",
#          "baseline":"arma"
#          }'

CATEGORIZATION_BASELINES_CONFIG='{
          "job": "categorization_baselines",
          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
          "users_steps_filename": "'$USERS_STEPS_FILENAME'",
          "ground_truth": "'$GROUND_TRUTH'",
          "base_dir":"'$GOWALLA_BASE_DIR'different_venues/",
          "adjacency_matrix_filename": "'$GOWALLA_ADJACENCY_FILENAME_US'",
           "temporal_matrix_filename": "'$GOWALLA_TEMPORAL_FILENAME_US'",
          "graph_type":"not_directed",
          "dataset_name":"gowalla",
          "country":"US",
          "state":"TEXAS",
          "categories_type":"7_categories",
          "baseline":"arma"
          }'

POI_CATEGORIZATION_VALIDATION='{
          "job": "poi_categorization_validation",
          "poi_categorized_filename": "'$POI_DETECTION_FILENAME'",
          "ground_truth": "'$GROUND_TRUTH'",
  }'

CATEGORIZATION_BASELINE_GPR_CONFIG='{
          "job": "poi_categorization_baseline_gpr",
          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
          "users_steps_filename": "'$USERS_STEPS_FILENAME'",
          "adjacency_matrix_filename": "'$GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_MONTHS_4_12_2012_US_ADJACENCY_MATRIX_FILENAME'",
          "feature_matrix_filename": "'$GLOBAL_FOURSQUARE_DIRECTED_GPR_8_CATEGORIES_MONTHS_4_12_2012_US_FEATURE_MATRIX_FILENAME'",
          "user_poi_vector_filename": "'$GLOBAL_FOURSQUARE_GPR_8_CATEGORIES_MONTHS_4_12_2012_US_SEQUENCES_MATRIX_FILENAME'",
          "user_metrics":"'$GLOBAL_FOURQUARE_USER_METRICS_FILENAME_US'",
          "graph_type":"directed",
          "dataset_name":"global_foursquare",
          "country":"US",
          "categories_type":"8_categories"
          }'

TRANSFER_LEARNING_CONFIG='{
          "job": "categorization",
          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
          "users_steps_filename": "'$USERS_STEPS_FILENAME'",
          "ground_truth": "'$RAW_GPS_GROUND_TRUTH'",
          "users_checkin_filename": "'$WEEPLACES_DATASET'",
          "adjacency_matrix_filename": "'$RAW_GPS_ADJACENCY_MATRIX_FILENAME'",
          "feature_matrix_filename": "'$RAW_GPS_FEATURE_MATRIX_FILENAME'",
          "user_metrics":"'$RAW_GPS_USER_METRICS_FILENAME'",
          "graph_type":"not_directed",
          "dataset_name":"weeplaces",
          "categories_type":"7_categories_osm"
          }'

CATEGORIZATION_PERFORMANCE_GRAPHICS_CONFIG='{
          "job": "categorization_performance_graphics",
          "dataset_name":"gowalla",
          "categories_type":"7_categories",
          "graph_type":"not_directed",
          "country":"US",
          "state":"TX",
          "version":"normal",
          "folds":"5",
          "replications":"1"
          }'

POI_TRANSACTIONS_ANALYSIS='{
          "job": "poi_transactions",
          "transactions_filename": "'$CHECK_INS_LOCAL_DATETIME_9_CATEGORIES_EVENT_EXCLUDED_FROM_2012_MONTH_4_TO_2013_MONTH_9_BR_US_JP_WITH_STATE_AND_CITY'",
          "transactions_base_dir":"'$TRANSACTIONS_BASE_DIR'",
          "base_dir":"'$GLOBAL_FOURSQUARE_BASE_DIR'matrizes/",
          "dataset_name":"global_foursquare",
          "generate_file":"no",
          "number_of_categories":"9_categories",
          "max_interval":"24",
          "different_venues":"yes",
          "country":"BR",
          "state":"sp",
          "county":"SÃO PAULO",
          }'


echo $CONFIG

case $1 in
  "categorization")
    python3 main.py "${CATEGORIZATION_CONFIG}"
    ;;
  "categorization_baselines")
    python3 main.py "${CATEGORIZATION_BASELINES_CONFIG}"
    ;;
  "categorization_performance")
    python3 main.py "${CATEGORIZATION_PERFORMANCE_GRAPHICS_CONFIG}"
    ;;
  "matrix_generation")
    python3 main.py "${MATRIX_GENERATION_FOR_POI_CATEGORIZATION_CONFIG}"
    ;;
  "hmrm_baseline")
    python3 main.py "${BASELINES_CONFIG}"
    ;;
  "poi_categorization validation")
    python3 main.py "${POI_CATEGORIZATION_VALIDATION}"
    ;;
  "categorization_gpr")
    python3 main.py "${CATEGORIZATION_BASELINE_GPR_CONFIG}"
    ;;
  "transfer")
    python3 main.py "${TRANSFER_LEARNING_CONFIG}"
    ;;
  "poi_transactions")
    python3 main.py "${POI_TRANSACTIONS_ANALYSIS}"
    ;;
esac