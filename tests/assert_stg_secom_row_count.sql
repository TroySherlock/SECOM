-- Fail when stg_secom row count differs from the UCI SECOM dataset (1567 observations).
select 1
from (
    select count(*) as n
    from {{ ref('stg_secom') }}
) t
where n != 1567
