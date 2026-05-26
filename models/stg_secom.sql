-- models/stg_secom.sql

with source as (

    select *
    from {{ ref('raw_secom') }}

),

cleaned as (

    select
        -- Remove embedded quote characters before timestamp parsing
        strptime(
            replace(timestamp, '"', ''),
            '%d/%m/%Y %H:%M:%S'
        ) as measurement_ts,

        * exclude (timestamp)

    from source

)

select *
from cleaned