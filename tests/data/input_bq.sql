-- Mock data for bigquery
CREATE SCHEMA IF NOT EXISTS `rinoa-core-prod.public`;
CREATE SCHEMA IF NOT EXISTS `rinoa-core-prod.temp_dataset`;
CREATE OR REPLACE TABLE `rinoa-core-prod.public.videos` AS (
  SELECT
    GENERATE_UUID() AS id,  -- Generate a UUID for each record
    CONCAT("Movie ", CAST(FLOOR(RAND() * 100000) AS STRING)) AS title,  -- Generate a random title
    CASE 
      WHEN RAND() < 0.2 THEN "mpa:pg-13"
      WHEN RAND() < 0.4 THEN "mpa:g"
      WHEN RAND() < 0.6 THEN "mpa:pg"
      WHEN RAND() < 0.8 THEN "mpa:r"
      ELSE "mpa:nc-17"
    END AS age_rating,  -- Randomly choose an age rating
    ARRAY(SELECT tag_list[OFFSET(CAST(5 * RAND() - 0.5 AS INT64))] 
      FROM UNNEST(GENERATE_ARRAY(1, 3))) AS tags
  FROM
    UNNEST(GENERATE_ARRAY(1, 30)),  -- Generate 10 records (modify the number as needed)
    (SELECT ['tag1','tag2','tag3','tag4','tag5'] tag_list)
);