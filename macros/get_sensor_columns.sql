{% macro get_sensor_columns(relation) %}
    {% set columns = adapter.get_columns_in_relation(relation) %}
    {% set sensor_cols = [] %}
    {% for col in columns %}
        {% if modules.re.match('^c_\\d+$', col.name) %}
            {% do sensor_cols.append(col.name) %}
        {% endif %}
    {% endfor %}
    {{ return(sensor_cols) }}
{% endmacro %}
