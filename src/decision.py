import numpy as np
# Evaluate total cost of each policy
def total_cost(order, actual, Cu, Co):
    shortage = np.maximum(actual - order, 0)
    excess = np.maximum(order - actual, 0)
    return (Cu * shortage + Co * excess).sum()

# Service level
def service_level(order, actual):
    return (order >= actual).mean()