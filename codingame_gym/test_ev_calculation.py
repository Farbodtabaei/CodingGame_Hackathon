"""Test script to debug EV calculation for turn 3 divergence"""

# V10-style EV calculation
def calc_ev(from_id, to_id, cost, path_length, turn=3):
    POINTS_PER_TURN = 3
    TOTAL_TURNS = 200
    my_bias_multiplier = -5.0
    
    # Same calculation as V12's calc_connection_ev_v10_style
    turns_to_complete = (cost + POINTS_PER_TURN - 1) // POINTS_PER_TURN
    turns_left = TOTAL_TURNS - turn
    scoring_turns = max(0, turns_left - turns_to_complete)
    
    if scoring_turns <= 0:
        return 0.0
    
    new_cells = path_length  # Assume all cells are new
    expected_points = new_cells * scoring_turns
    
    # V10's speed multiplier
    speed_multiplier = 1.5 if cost <= 10 else (1.2 if cost <= 20 else 0.9)
    weighted_points = expected_points * speed_multiplier
    
    # Efficiency bonus
    efficiency = weighted_points / max(cost, 1)
    
    # Town ID bonus
    town_id_bonus = 0
    if from_id > 0 and to_id > 0:
        base_bonus = (from_id + to_id) * 50
        town_id_bonus = base_bonus * my_bias_multiplier
    
    ev = weighted_points + efficiency * 100 + town_id_bonus
    
    print(f"\nConnection {from_id}->{to_id}:")
    print(f"  cost={cost}, path_length={path_length}")
    print(f"  turns_to_complete={turns_to_complete}, scoring_turns={scoring_turns}")
    print(f"  expected_points={expected_points}")
    print(f"  speed_multiplier={speed_multiplier}, weighted_points={weighted_points:.1f}")
    print(f"  efficiency={efficiency:.1f}")
    print(f"  town_id_sum={from_id + to_id}, town_id_bonus={town_id_bonus:.1f}")
    print(f"  TOTAL EV={ev:.1f}")
    
    return ev

# From the logs:
# V12 chose 6->2 with path_length=7
# V10 chose 4->1 with path_length=8

# Need to estimate costs - let's use the path lengths as proxy
# In plains terrain (cost=1 per cell), cost ~= path_length
print("=" * 60)
print("TURN 3 CONNECTION COMPARISON")
print("=" * 60)

# Test both connections
ev_6_2 = calc_ev(6, 2, 7, 7, turn=3)
ev_4_1 = calc_ev(4, 1, 8, 8, turn=3)

print("\n" + "=" * 60)
print(f"Connection 6->2 EV: {ev_6_2:.1f}")
print(f"Connection 4->1 EV: {ev_4_1:.1f}")
print(f"V12 chose: 6->2 (EV={ev_6_2:.1f})")
print(f"V10 chose: 4->1 (EV={ev_4_1:.1f})")
print("=" * 60)
