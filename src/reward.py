def calculate_reward(units_sold, units_in_stock, demand, unit_price):

    holding_cost_per_unit = 0.10  
    stockout_penalty_per_unit = unit_price * 1.5 
    margin_per_unit = unit_price * 0.3 

    revenue_gain = units_sold * margin_per_unit
    
    holding_loss = units_in_stock * holding_cost_per_unit
    
    missed_sales = max(0, demand - units_sold)
    stockout_loss = missed_sales * stockout_penalty_per_unit
    
    reward = revenue_gain - holding_loss - stockout_loss
    return reward