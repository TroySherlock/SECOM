-- int_secom_features.sql

{% set cols = adapter.get_columns_in_relation(ref('stg_secom')) %}
{% set sensor_cols = [] %}

{% for col in cols %}
    {% if col.name not in ['measurement_ts', 'target', 'year'] %}
        {% do sensor_cols.append(col.name) %}
    {% endif %}
{% endfor %}

with base as (

    select *
    from {{ ref('stg_secom') }}

),

-- Step 1: Remove irrelevant or constant columns
filtered as (
    select
        measurement_ts,
        target,
        {% for col in sensor_cols %}
            {{ col }}{% if not loop.last %},{% endif %}
        {% endfor %}
    from base
),

-- Step 2: Extract time components
time_features as (
    select
        *,
        extract(month from measurement_ts) as month,
        extract(dow from measurement_ts) as day_of_week,  -- 0=Sunday, 6=Saturday
        extract(hour from measurement_ts) as hour,
        case when extract(dow from measurement_ts) in (0,6) then 1 else 0 end as is_weekend
    from filtered
),

-- Step 3: Cyclical encoding (n_missing_sensors added in mart after profiling)
cyclical as (
    select
        *,
        sin(2 * pi() * month / 12) as month_sin,
        cos(2 * pi() * month / 12) as month_cos,
        sin(2 * pi() * day_of_week / 7) as dow_sin,
        cos(2 * pi() * day_of_week / 7) as dow_cos,
        sin(2 * pi() * hour / 24) as hour_sin,
        cos(2 * pi() * hour / 24) as hour_cos
    from time_features
)

select * from cyclical