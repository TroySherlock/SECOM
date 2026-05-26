{% macro get_missing_indicator_sensors() %}
    {% set query %}
        select column_name
        from {{ ref('int_secom_column_metadata') }}
        where keep_in_mart and has_missing
        order by column_name
    {% endset %}

    {% if execute %}
        {% set results = run_query(query) %}
        {% if results is none or results.rows | length == 0 %}
            {{ return([]) }}
        {% else %}
            {{ return(results.columns[0].values() | list) }}
        {% endif %}
    {% else %}
        {{ return([]) }}
    {% endif %}
{% endmacro %}
