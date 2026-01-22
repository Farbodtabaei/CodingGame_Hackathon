import jpype
import json

from typing import List

from codingame_gym.cggym import start_jvm, envs_catalog

class GameRunner:
    """
    A class for running games between two agents using the java game engine.

    Args:
        env_name (str): The name of the environment.

    Raises:
        ValueError: If the specified game environment is not found in the catalog.
        Exception: If the GameMain java class cannot be found. GameMain is the main class for running games in the java game engine.
    """

    def __init__(self, env_name):
        """
        Initializes the GameRunner object.

        Args:
            env_name (str): The name of the environment.

        Raises:
            Exception: If the GameMain class cannot be found.
        """
        if env_name not in envs_catalog:
            raise ValueError(f"Game '{env_name}' not found in catalog. Use list_envs() to get a list of available games.")
    
        if not jpype.isJVMStarted():
            start_jvm(envs_catalog[env_name]['jar_path'])

        try:
            from com.codingame.main import GameMain
            self.GameMain = GameMain
        except Exception as e:
            raise Exception('Could not find GameMain class. Report the bug to fzinno@ea.com.\nOriginal exception: ' + str(e))

    def run_games(self, num_games:int, league_level:int, 
                agent_1_cmd:str, agent_2_cmd:str, 
                seeds:List[int] = None, debug:bool = False):
        """
        Run a series of games between two agents in parallel running the java game engine.
        Due to the java API, seed and league_level will be used for all the games.
        If you want to run games with different seeds or league levels, you will have to call this method multiple times.
        The agent commands can be with any language the CG engine supports, in other works, the cmd will be launched as separate processes.

        Args:
            num_games (int): The number of games to run.
            league_level (int): The league level of the games.
            agent_1_cmd (str): The cmd for the first agent, e.g "python agent1.py".
            agent_2_cmd (str): The name of the second agent, e.g "python agent2.py".
            seed (int, optional): The seed for random number generation. Defaults to None.
            debug (bool, optional): Whether to run the games in debug mode. Debug mode will serve a replay of the game on localhost and will only run one game. Defaults to False.

        Returns:
            list: A list of scores for each game, or None if debug is True.

        Raises:
            Exception: If the games cannot be run.
        """

        # Check for incompatible arguments
        effective_num_games = num_games
        if debug and num_games > 1:
            effective_num_games = 1
            print('Warning: cannot run multiple games in debug mode. Only one game will be run.')

        if len(seeds) != num_games:
            raise ValueError('The number of seeds must match the number of games.')

        args = [
            f'-n={effective_num_games}',
            f'-l={league_level}',
            '-t', 
            '-j', 
            '-p', agent_1_cmd, agent_2_cmd
        ]

        if seeds:
            for seed in seeds:
                args += ['-s', str(seed)]
        if debug:
            args += ['-d']
        if effective_num_games > 1:
            args += ['-c']

        results = ''
        try:
            results = self.GameMain.runGames(args)
        except Exception as e:
            print('Could not run games. Report the bug to fzinno@ea.com.')
            raise e
        
        if not results and not debug:
            raise Exception('Could not get games results. Report the bug to fzinno@ea.com.')
        
        if results:
            results_json = json.loads(str(results))
            scores = [game['scores'] for game in results_json['results']]
            return scores
        else:
            return None
