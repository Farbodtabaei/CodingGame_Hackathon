"""
Arena testing system - All strategic bots compete against each other.
Round-robin tournament with detailed statistics.
"""
import sys
from pathlib import Path
import importlib.util
import builtins
import random
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import codingame_gym.cggym as gym


def load_bot(bot_path, module_name, class_name):
    """Dynamically load a bot module."""
    spec = importlib.util.spec_from_file_location(module_name, bot_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def run_match(bot1_class, bot2_class, seed=None, verbose=False):
    """Run a single match between two bots. Returns (score1, score2)."""
    env = gym.make("backtrack", num_envs=1, debug=True)
    env.reset(league_level=3, random_seeds=[seed or random.randint(0, 999999)])

    # Get init info
    raw_infos = env.env.global_info()
    raw_info_p0 = [str(item) for item in raw_infos[0]]
    
    # Create player 1's init info (swap player ID from 0 to 1)
    raw_info_p1 = raw_info_p0.copy()
    raw_info_p1[0] = '1'
    
    def fake_input_gen(lines):
        it = iter(lines)
        return lambda: next(it)
    
    orig_input = builtins.input

    # Initialize bots
    builtins.input = fake_input_gen(raw_info_p0)
    bot1 = bot1_class()
    bot1.read_init()
    
    builtins.input = fake_input_gen(raw_info_p1)
    bot2 = bot2_class()
    bot2.read_init()
    
    builtins.input = orig_input

    width = bot1.width
    height = bot1.height
    turn = 0
    done = False
    
    while not done and turn < 100:
        turn += 1
        
        # Get per-player frame strings
        frame_strings = env.get_frame_strings()
        
        if len(frame_strings) >= 2:
            turn_lines_p0 = [str(line) for line in frame_strings[0]]
            turn_lines_p1 = [str(line) for line in frame_strings[1]]
        else:
            scores = env.get_scores()
            score1 = int(float(scores[0]))
            score2 = int(float(scores[1]))
            turn_lines_p0 = [str(score1), str(score2)]
            turn_lines_p1 = [str(score2), str(score1)]
            
            raw_obs = env.observations
            if isinstance(raw_obs, (list, tuple)):
                raw_obs = raw_obs[0]
            
            features_per_cell = 4
            for y in range(height):
                for x in range(width):
                    idx = (y * width + x) * features_per_cell
                    if idx + 3 < len(raw_obs):
                        owner = int(raw_obs[idx])
                        instability = int(raw_obs[idx + 1])
                        inked = int(raw_obs[idx + 2])
                        connections = raw_obs[idx + 3]
                        inked_str = "1" if inked else "0"
                        conn_str = str(int(connections)) if isinstance(connections, (int, float)) else str(connections)
                        turn_lines_p0.append(f"{owner} {instability} {inked_str} {conn_str}")
                        owner_p1 = 1 if owner == 0 else (0 if owner == 1 else owner)
                        turn_lines_p1.append(f"{owner_p1} {instability} {inked_str} {conn_str}")
                    else:
                        turn_lines_p0.append("-1 0 0 x")
                        turn_lines_p1.append("-1 0 0 x")
        
        # Get actions
        builtins.input = fake_input_gen(turn_lines_p0)
        bot1.read_turn()
        action1 = bot1.get_action()
        
        builtins.input = fake_input_gen(turn_lines_p1)
        bot2.read_turn()
        action2 = bot2.get_action()
        
        builtins.input = orig_input
        
        # Execute
        actions = [[action1], [action2]]
        result = env.step(actions)
        
        scores_after = env.get_scores()
        score1_after = int(float(scores_after[0]))
        score2_after = int(float(scores_after[1]))
        
        if verbose and turn <= 20:
            print(f"Turn {turn:3d} | Score: {score1_after:4d} vs {score2_after:4d} | {action1[:40]:40s} | {action2[:40]:40s}")
        
        if isinstance(result, tuple):
            if len(result) == 4:
                obs, reward, done_flag, info = result
            elif len(result) == 5:
                obs, reward, done_flag, truncated, info = result
            else:
                done_flag = result[0] if result else False
        else:
            done_flag = result
        
        if isinstance(done_flag, (list, tuple)):
            done = done_flag[0]
        else:
            done = done_flag
    
    # Final scores
    final_scores = env.get_scores()
    final1 = int(float(final_scores[0]))
    final2 = int(float(final_scores[1]))
    
    if hasattr(env, 'close'):
        env.close()
    
    return final1, final2


def run_tournament(bots, matches_per_pair=50):
    """
    Run round-robin tournament between all bots.
    
    Args:
        bots: dict of {name: bot_class}
        matches_per_pair: number of matches per pairing (each position)
    
    Returns:
        dict of statistics per bot
    """
    bot_names = list(bots.keys())
    n_bots = len(bot_names)
    
    # Stats tracking
    stats = {name: {
        'wins': 0,
        'losses': 0,
        'ties': 0,
        'total_score': 0,
        'total_opponent_score': 0,
        'matches_played': 0,
        'matchups': defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0})
    } for name in bot_names}
    
    print("=" * 80)
    print(f"ARENA TOURNAMENT: {n_bots} bots, {matches_per_pair} matches per pairing")
    print("=" * 80)
    print()
    
    # Round-robin: each bot plays each other bot
    total_pairings = n_bots * (n_bots - 1) // 2
    pairing_num = 0
    
    for i in range(n_bots):
        for j in range(i + 1, n_bots):
            pairing_num += 1
            bot1_name = bot_names[i]
            bot2_name = bot_names[j]
            bot1_class = bots[bot1_name]
            bot2_class = bots[bot2_name]
            
            print(f"[{pairing_num}/{total_pairings}] {bot1_name} vs {bot2_name} ({matches_per_pair} matches each position)...")
            
            b1_wins = 0
            b2_wins = 0
            ties = 0
            b1_total_score = 0
            b2_total_score = 0
            
            # Play matches with bot1 as P0
            for k in range(matches_per_pair):
                seed = 10000 * (i * 100 + j) + k * 7919
                score1, score2 = run_match(bot1_class, bot2_class, seed=seed, verbose=False)
                
                b1_total_score += score1
                b2_total_score += score2
                
                if score1 > score2:
                    b1_wins += 1
                elif score2 > score1:
                    b2_wins += 1
                else:
                    ties += 1
            
            # Play matches with bot2 as P0
            for k in range(matches_per_pair):
                seed = 50000 * (i * 100 + j) + k * 7919
                score2, score1 = run_match(bot2_class, bot1_class, seed=seed, verbose=False)
                
                b1_total_score += score1
                b2_total_score += score2
                
                if score1 > score2:
                    b1_wins += 1
                elif score2 > score1:
                    b2_wins += 1
                else:
                    ties += 1
            
            total_matches = matches_per_pair * 2
            
            # Update stats
            stats[bot1_name]['wins'] += b1_wins
            stats[bot1_name]['losses'] += b2_wins
            stats[bot1_name]['ties'] += ties
            stats[bot1_name]['total_score'] += b1_total_score
            stats[bot1_name]['total_opponent_score'] += b2_total_score
            stats[bot1_name]['matches_played'] += total_matches
            stats[bot1_name]['matchups'][bot2_name]['wins'] = b1_wins
            stats[bot1_name]['matchups'][bot2_name]['losses'] = b2_wins
            stats[bot1_name]['matchups'][bot2_name]['ties'] = ties
            
            stats[bot2_name]['wins'] += b2_wins
            stats[bot2_name]['losses'] += b1_wins
            stats[bot2_name]['ties'] += ties
            stats[bot2_name]['total_score'] += b2_total_score
            stats[bot2_name]['total_opponent_score'] += b1_total_score
            stats[bot2_name]['matches_played'] += total_matches
            stats[bot2_name]['matchups'][bot1_name]['wins'] = b2_wins
            stats[bot2_name]['matchups'][bot1_name]['losses'] = b1_wins
            stats[bot2_name]['matchups'][bot1_name]['ties'] = ties
            
            print(f"  Result: {bot1_name} {b1_wins}W-{b2_wins}L-{ties}T | " +
                  f"Avg scores: {bot1_name} {b1_total_score/total_matches:.0f} vs " +
                  f"{bot2_name} {b2_total_score/total_matches:.0f}")
            print()
    
    return stats


def print_standings(stats):
    """Print tournament standings."""
    print()
    print("=" * 80)
    print("FINAL STANDINGS")
    print("=" * 80)
    print()
    
    # Calculate win rates and sort
    rankings = []
    for name, s in stats.items():
        total = s['matches_played']
        win_rate = s['wins'] / total if total > 0 else 0
        avg_score = s['total_score'] / total if total > 0 else 0
        avg_opp_score = s['total_opponent_score'] / total if total > 0 else 0
        score_diff = avg_score - avg_opp_score
        
        rankings.append({
            'name': name,
            'wins': s['wins'],
            'losses': s['losses'],
            'ties': s['ties'],
            'win_rate': win_rate,
            'avg_score': avg_score,
            'avg_opp_score': avg_opp_score,
            'score_diff': score_diff,
            'total_matches': total
        })
    
    # Sort by win rate, then score differential
    rankings.sort(key=lambda x: (x['win_rate'], x['score_diff']), reverse=True)
    
    # Print table
    print(f"{'Rank':<6} {'Bot':<12} {'W-L-T':<15} {'Win %':<8} {'Avg Score':<12} {'Opp Score':<12} {'Diff':<8}")
    print("-" * 80)
    
    for rank, r in enumerate(rankings, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{emoji} #{rank:<4} {r['name']:<12} {r['wins']:>3}-{r['losses']:<3}-{r['ties']:<3}   " +
              f"{r['win_rate']*100:>5.1f}%  {r['avg_score']:>8.0f}     {r['avg_opp_score']:>8.0f}     " +
              f"{r['score_diff']:>+6.0f}")
    
    print()
    print("=" * 80)
    print("HEAD-TO-HEAD MATCHUPS")
    print("=" * 80)
    print()
    
    # Print matchup matrix
    bot_names = [r['name'] for r in rankings]
    
    for i, bot1 in enumerate(bot_names):
        print(f"\n{bot1}:")
        for bot2 in bot_names:
            if bot1 == bot2:
                continue
            matchup = stats[bot1]['matchups'][bot2]
            w = matchup['wins']
            l = matchup['losses']
            t = matchup['ties']
            total = w + l + t
            wr = w / total * 100 if total > 0 else 0
            print(f"  vs {bot2:<12}: {w:>3}W-{l:<3}L-{t:<3}T ({wr:>5.1f}%)")


def test_single_bot_vs_all(target_version, matches_per_pair=10):
    """
    Test a single bot against all other versions.
    
    Args:
        target_version: Version number to test (e.g., 10 for V10)
        matches_per_pair: Matches per position (default 10, 20 total per opponent)
    """
    # Load target bot
    target_bot_path = PROJECT_ROOT / f"strategic_bot_v{target_version}.py"
    if not target_bot_path.exists():
        print(f"❌ V{target_version} not found at {target_bot_path}")
        return
    
    try:
        if target_version <= 7:
            target_class = load_bot(str(target_bot_path), f"strategic_bot_v{target_version}", "StrategicBot")
        else:
            target_class = load_bot(str(target_bot_path), f"strategic_bot_v{target_version}", f"StrategicBotV{target_version}")
        print(f"✓ Loaded V{target_version} (target)")
    except Exception as e:
        print(f"❌ Failed to load V{target_version}: {e}")
        return
    
    # Load all other bots
    opponents = {}
    for version in range(2, 11):
        if version == target_version:
            continue
        bot_path = PROJECT_ROOT / f"strategic_bot_v{version}.py"
        if bot_path.exists():
            try:
                if version <= 7:
                    bot_class = load_bot(str(bot_path), f"strategic_bot_v{version}", "StrategicBot")
                else:
                    bot_class = load_bot(str(bot_path), f"strategic_bot_v{version}", f"StrategicBotV{version}")
                opponents[f"V{version}"] = bot_class
                print(f"✓ Loaded V{version}")
            except Exception as e:
                print(f"✗ Failed to load V{version}: {e}")
    
    if len(opponents) == 0:
        print(f"\n❌ No opponent bots found!")
        return
    
    print(f"\n{len(opponents)} opponents ready!\n")
    
    # Test against each opponent
    print("=" * 80)
    print(f"V{target_version} vs ALL OPPONENTS ({matches_per_pair} matches per position)")
    print("=" * 80)
    print()
    
    total_wins = 0
    total_losses = 0
    total_ties = 0
    total_score = 0
    total_opp_score = 0
    total_matches = 0
    
    results = []
    
    for opp_name, opp_class in sorted(opponents.items()):
        print(f"Testing V{target_version} vs {opp_name}...")
        
        wins = 0
        losses = 0
        ties = 0
        my_score = 0
        opp_score = 0
        
        # Test as P0
        for i in range(matches_per_pair):
            seed = 10000 * target_version + i * 7919
            s1, s2 = run_match(target_class, opp_class, seed=seed, verbose=False)
            my_score += s1
            opp_score += s2
            if s1 > s2:
                wins += 1
            elif s2 > s1:
                losses += 1
            else:
                ties += 1
        
        # Test as P1
        for i in range(matches_per_pair):
            seed = 50000 * target_version + i * 7919
            s2, s1 = run_match(opp_class, target_class, seed=seed, verbose=False)
            my_score += s1
            opp_score += s2
            if s1 > s2:
                wins += 1
            elif s2 > s1:
                losses += 1
            else:
                ties += 1
        
        match_total = matches_per_pair * 2
        win_rate = wins / match_total * 100
        avg_score = my_score / match_total
        avg_opp = opp_score / match_total
        
        total_wins += wins
        total_losses += losses
        total_ties += ties
        total_score += my_score
        total_opp_score += opp_score
        total_matches += match_total
        
        status = "✅" if wins > losses else "⚠️" if wins == losses else "❌"
        results.append({
            'opponent': opp_name,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_rate': win_rate,
            'avg_score': avg_score,
            'avg_opp': avg_opp,
            'status': status
        })
        
        print(f"  {status} {wins}W-{losses}L-{ties}T ({win_rate:.1f}%) | " +
              f"Avg: V{target_version} {avg_score:.0f} vs {opp_name} {avg_opp:.0f}")
        print()
    
    # Summary
    print("=" * 80)
    print(f"V{target_version} OVERALL RESULTS")
    print("=" * 80)
    print()
    print(f"Total Matches: {total_matches}")
    print(f"Total Wins:    {total_wins} ({total_wins/total_matches*100:.1f}%)")
    print(f"Total Losses:  {total_losses} ({total_losses/total_matches*100:.1f}%)")
    print(f"Total Ties:    {total_ties} ({total_ties/total_matches*100:.1f}%)")
    print()
    print(f"Average Score:     {total_score/total_matches:.0f}")
    print(f"Opponent Avg:      {total_opp_score/total_matches:.0f}")
    print(f"Score Differential: {(total_score - total_opp_score)/total_matches:+.0f}")
    print()
    
    if total_wins > total_losses:
        print(f"🏆 V{target_version} DOMINATES! (+{total_wins - total_losses} net wins)")
    elif total_wins == total_losses:
        print(f"⚖️  V{target_version} EVEN with opponents")
    else:
        print(f"❌ V{target_version} needs improvement (-{total_losses - total_wins} net losses)")
    print()
    
    # Detailed table
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print()
    print(f"{'Opponent':<10} {'W-L-T':<15} {'Win %':<10} {'Avg Score':<12} {'Opp Score':<12} {'Status':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['opponent']:<10} {r['wins']:>3}-{r['losses']:<3}-{r['ties']:<3}    " +
              f"{r['win_rate']:>6.1f}%   {r['avg_score']:>8.0f}     {r['avg_opp']:>8.0f}      {r['status']}")


def main():
    import sys
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        # Single bot testing mode
        try:
            target_version = int(sys.argv[1])
            matches = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            test_single_bot_vs_all(target_version, matches_per_pair=matches)
        except ValueError:
            print("Usage: python test_arena.py <version> [matches_per_pair]")
            print("Example: python test_arena.py 10 50")
        return
    
    # Original tournament mode - discover all bots
    bots = {}
    
    for version in range(7, 11):  # V7 through V10
        bot_path = PROJECT_ROOT / f"strategic_bot_v{version}.py"
        if bot_path.exists():
            try:
                if version <= 7:
                    bot_class = load_bot(str(bot_path), f"strategic_bot_v{version}", "StrategicBot")
                else:
                    bot_class = load_bot(str(bot_path), f"strategic_bot_v{version}", f"StrategicBotV{version}")
                bots[f"V{version}"] = bot_class
                print(f"✓ Loaded V{version}")
            except Exception as e:
                print(f"✗ Failed to load V{version}: {e}")
    
    if len(bots) < 2:
        print(f"\nNeed at least 2 bots to run tournament. Found: {len(bots)}")
        return
    
    print(f"\n{len(bots)} bots ready for tournament!\n")
    
    # Run tournament
    matches_per_pair = 30  # 30 matches per position = 60 total per pairing
    stats = run_tournament(bots, matches_per_pair=matches_per_pair)
    
    # Print results
    print_standings(stats)


if __name__ == "__main__":
    main()
