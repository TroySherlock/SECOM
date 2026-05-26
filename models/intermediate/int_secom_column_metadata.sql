-- depends_on: {{ ref('int_secom_features') }}

-- Profile c_* sensors from int_secom_features for mart keep/drop rules.
-- Run: dbt run -s +mart_secom_features
-- Note: DuckDB UNPIVOT drops nulls, so stats use per-sensor UNION ALL (one table scan per sensor).

{{ config(materialized='table') }}

{% if execute %}
    {% set sensor_cols = get_sensor_columns(ref('int_secom_features')) %}
{% else %}
    {% set sensor_cols = [] %}
{% endif %}

{% if execute and sensor_cols | length == 0 %}
    {{ exceptions.raise_compiler_error("No sensor columns matching ^c_\\d+$ in int_secom_features.") }}
{% endif %}

with profiled as (

    {% if execute %}
        {% for col in sensor_cols %}
        select
            '{{ col }}' as column_name,
            count(*) as n_rows,
            count(*) - count({{ col }}) as n_null,
            (count(*) - count({{ col }}))::double / count(*) as missing_rate,
            count(distinct {{ col }}) as n_distinct,
            var_pop({{ col }}) as raw_variance
        from {{ ref('int_secom_features') }}
        {% if not loop.last %}
        union all
        {% endif %}
        {% endfor %}
    {% else %}
        select
            cast(null as varchar) as column_name,
            cast(null as bigint) as n_rows,
            cast(null as bigint) as n_null,
            cast(null as double) as missing_rate,
            cast(null as bigint) as n_distinct,
            cast(null as double) as raw_variance
        where false
    {% endif %}

),

final as (

    select
        column_name,
        n_rows,
        n_null,
        missing_rate,
        n_distinct,
        case
            when raw_variance is null or raw_variance = 0 then 0.0
            else raw_variance
        end as variance,
        n_null > 0 as has_missing,
        missing_rate < 0.10
            and not (raw_variance is null or raw_variance = 0) as keep_in_mart,
        case
            when missing_rate >= 0.10 then 'high_missing'
            when raw_variance is null or raw_variance = 0 then 'zero_variance'
            else 'ok'
        end as drop_reason,
        current_timestamp as profiled_at,
        'int_secom_features' as source_relation
    from profiled

)

select *
from final
order by column_name
