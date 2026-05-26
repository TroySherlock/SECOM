{% macro get_deduped_missing_indicator_sensors() %}
    {# One c_*__missing column per identical null pattern across int_secom_features rows. #}
    {% set candidates = get_missing_indicator_sensors() %}
    {% if not execute or candidates | length == 0 %}
        {{ return([]) }}
    {% endif %}

    {% set union_parts = [] %}
    {% for col in candidates %}
        {% set part %}
            select
                '{{ col }}' as column_name,
                md5(string_agg(
                    case when {{ col }} is null then '1' else '0' end,
                    '' order by measurement_ts
                )) as pattern_hash
            from {{ ref('int_secom_features') }}
        {% endset %}
        {% do union_parts.append(part) %}
    {% endfor %}

    {% set query %}
        with per_sensor as (
            {{ union_parts | join('\n            union all\n') }}
        ),
        representatives as (
            select column_name
            from per_sensor
            qualify row_number() over (
                partition by pattern_hash
                order by column_name
            ) = 1
        )
        select column_name
        from representatives
        order by column_name
    {% endset %}

    {% set results = run_query(query) %}
    {% if results is none or results.rows | length == 0 %}
        {{ return([]) }}
    {% else %}
        {{ return(results.columns[0].values() | list) }}
    {% endif %}
{% endmacro %}
