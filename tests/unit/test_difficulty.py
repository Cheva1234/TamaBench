from tamabench.env.scenarios import get_difficulty_config
from tamabench.env.economy import EconomySystem

def test_dynamic_difficulty_scaling():
    # Day 0
    jobs_day0 = EconomySystem.get_dynamic_jobs(0)
    shop_day0 = EconomySystem.get_dynamic_shop(0)
    
    assert jobs_day0[0].reward == 60 # cafe_shift
    assert shop_day0[0].cost == 10 # food
    
    # Day 10 (14400 minutes)
    jobs_day10 = EconomySystem.get_dynamic_jobs(14400)
    shop_day10 = EconomySystem.get_dynamic_shop(14400)
    
    # Value approaches limits
    assert jobs_day10[0].reward < 60
    assert shop_day10[0].cost > 10
