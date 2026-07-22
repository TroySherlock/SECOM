-- models/marts/mart_secom_features.sql
-- Run: dbt run -s +mart_secom_features

{% set kept_query %}
    select column_name
    from {{ ref('int_secom_column_metadata') }}
    where keep_in_mart
    order by column_name
{% endset %}

{% if execute %}
    {% set kept_results = run_query(kept_query) %}

    {% if kept_results is none or kept_results.rows | length == 0 %}
        {{ exceptions.raise_compiler_error(
            "No kept sensors in int_secom_column_metadata. "
            "Run: dbt run -s int_secom_column_metadata mart_secom_features"
        ) }}
    {% endif %}

    {% set kept_sensors = kept_results.columns[0].values() | list %}
    {% set missing_indicator_sensors = get_deduped_missing_indicator_sensors() %}
{% else %}
    {% set kept_sensors = [] %}
    {% set missing_indicator_sensors = [] %}
{% endif %}

with features as (

    select *
    from {{ ref('int_secom_features') }}

),

selected as (

    select
        raw_row_id,
        measurement_ts,
        target,
        is_weekend,
        month_sin,
        month_cos,
        dow_sin,
        dow_cos,
        hour_sin,
        hour_cos
        {% if kept_sensors | length > 0 %}
        ,
        {% for col in kept_sensors %}
        {{ col }}{% if not loop.last %},{% endif %}
        {% endfor %}
        {% endif %}
    from features

),

-- Causal rolling z-score (local mean / sample-SD standardization) for the
-- extrapolation track. Strictly-past window (excludes the current row) so no
-- future leakage; the raw absolutes are retained and rz columns are added
-- alongside them. raw_row_id breaks ties between rows sharing a timestamp so
-- the window (and observation_id below) is deterministic across runs.
{% set rz_window_rows = 50 %}
rolled as (

    select
        *
        {% if kept_sensors | length > 0 %}
        {% for col in kept_sensors %}
        ,
        coalesce(
            ({{ col }} - avg({{ col }}) over rz_w)
              / nullif(stddev_samp({{ col }}) over rz_w, 0),
            0
        ) as {{ col }}_rz
        {% endfor %}
        {% endif %}
    from selected
    window rz_w as (order by measurement_ts, raw_row_id rows between {{ rz_window_rows }} preceding and 1 preceding)

),

enriched as (

    select
        *
        {% if missing_indicator_sensors | length > 0 %}
        ,
        {% for col in missing_indicator_sensors %}
        cast(case when {{ col }} is null then 1 else 0 end as integer) as {{ col }}__missing{% if not loop.last %},{% endif %}
        {% endfor %}
        {% endif %}
        ,
        (
            {% if kept_sensors | length > 0 %}
            {% for col in kept_sensors %}
            cast(case when {{ col }} is null then 1 else 0 end as integer){% if not loop.last %} + {% endif %}
            {% endfor %}
            {% else %}
            0
            {% endif %}
        ) as n_missing_sensors
    from rolled

),

final as (

    select
        row_number() over (order by measurement_ts, raw_row_id) as observation_id,
        *
    from enriched

)

select * exclude (raw_row_id)
from final
