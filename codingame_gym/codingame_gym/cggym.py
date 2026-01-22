import jpype
import jpype.imports
import jpype.nio
import sys
import os
import threading
import numpy as np

from codingame_gym.utils import start_jvm, convert_to_java_list_of_lists, convert_to_java_list_of_lists_of_lists
from codingame_gym.catalogue import envs_catalog
from codingame_gym.utils import RedirectJavaOutput


class GymError(Exception):
    """Custom exception for gym errors that should be displayed cleanly."""
    pass


def _custom_excepthook(exc_type, exc_value, exc_traceback):
    """Custom exception handler to print clean error messages for GymError."""
    if exc_type == GymError:
        print(f"Error: {exc_value}", file=sys.stderr)
        sys.exit(1)
    else:
        # For other exceptions, use default handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


# Install custom exception hook
sys.excepthook = _custom_excepthook


def list_envs():
    """
    Returns a list of available environment names.

    Returns:
        list: A list of environment names.
    """
    return list(envs_catalog.keys())

def make(env_name:str, num_envs:int=1, debug:bool=False, num_threads:int=None):
    """
    Creates and returns a game environment based on the specified environment name.
    ByteBuffer shared memory is always enabled for efficient zero-copy observations.

    Args:
        env_name (str): The name of the game environment.
        num_envs (int): Number of parallel environments (default=1 for single environment).
        debug (bool): Enable debug mode for string observation capture (default=False).
                     WARNING: Debug mode adds ~15-20% performance overhead.
        num_threads (int): Number of threads for parallel execution (default=None, uses CPU count * 2).
                          Only relevant when num_envs > 1.

    Returns:
        game_env: The created game environment.

    Raises:
        GymError: If the specified environment name is not found in the catalog.
        Exception: If there is an error starting the JVM or finding the GymEnv class.
    
    Example:
        >>> import codingame_gym.cggym as gym
        >>> env = gym.make('backtrack', num_envs=10, num_threads=8)
        >>> env.reset(league_level=3, random_seed=12345)
        >>> # Access numeric observations
        >>> obs = env.observations  # shape: [num_players, obs_size]
        >>> # Access string observations (debug mode only)
        >>> strings = env.get_frame_strings()  # List[List[str]]
    """
    if env_name not in envs_catalog:
        raise GymError(f"Game '{env_name}' not found in catalog. Use list_envs() to get a list of available games.")
    
    try:
        start_jvm(envs_catalog[env_name]['jar_path'])
    except Exception as e:
        print('Could not start JVM. Report the bug to fzinno@ea.com.')
        raise e

    
    try:
        from com.codingame.game import GymEnv as JavaGymEnv
    except Exception as e:
        print('Could not find GymEnv class. Report the bug to fzinno@ea.com.')
        raise e

    with RedirectJavaOutput():
        game_env = GameEnv.__new__(GameEnv)
        game_env._reset_called = False
        game_env.num_envs = num_envs
        game_env.debug_mode = debug
        if num_threads is None:
            game_env.env = JavaGymEnv.make(num_envs, debug)
        else:
            game_env.env = JavaGymEnv.make(num_envs, debug, num_threads)

    return game_env


def convert_actions_to_string(actions):
    """
    Convert a List[List[List[str]]] to a single string with specific separators.

    Args:
        actions (list): A nested list where outer lists are episodes, inner lists are players, and innermost lists are actions.

    Returns:
        str: A formatted string where episodes are separated by ';', players by ',', and actions by '\n'.
    """
    episodes = []
    for episode in actions:
        players = []
        for player in episode:
            players.append("\n".join(player))
        episodes.append(",".join(players))
    return ";".join(episodes)

class GameEnv:
    """
    Unified game environment that handles both single and multi-environment modes.
    
    The behavior adapts based on num_envs:
    - num_envs=1: Single environment mode (original GameEnv behavior)
    - num_envs>1: Multi environment mode (original MultiGameEnv behavior)

    Attributes:
        env: The underlying Java game environment.
        num_envs: Number of parallel environments.
        shared_buffers: List of ByteBuffers for zero-copy observation reading (if enabled).
        shared_arrays: Cached numpy arrays mapped to ByteBuffers (if enabled).
    """
    
    def _setup_shared_buffers(self):
        """
        Setup shared ByteBuffer access for zero-copy observation reading.
        Observations are always written to ByteBuffer by StateEncoder.
        """
        try:
            # Get the shared ByteBuffer via the getSharedBuffer() method
            shared_buffer = self.env.getSharedBuffer()
            
            # Convert to Python buffer
            py_buffer = jpype.nio.convertToDirectBuffer(shared_buffer)
            
            # Read constants from StateEncoder.java (single source of truth)
            from com.codingame.game import StateEncoder
            self.obs_size = StateEncoder.OBS_SIZE
            self.num_players = 2
            
            # Create numpy array view for ALL environments
            total_floats = self.num_envs * self.num_players * self.obs_size
            self.shared_array = np.frombuffer(py_buffer, dtype=np.float32, count=total_floats)
            self.shared_array = self.shared_array.reshape(self.num_envs, self.num_players, self.obs_size)
            
            self._shared_buffer_ready = True
            
        except Exception as e:
            raise GymError(f"Failed to setup ByteBuffer shared memory: {e}") from None

    def get_player_count(self):
        """
        Get the number of players in the game.

        Returns:
            The number of players.
        """
        return self.env.get_player_count()
    
    @property
    def observations(self):
        """
        Direct access to shared observations buffer (zero-copy).
        
        Returns:
            Single env (num_envs=1): numpy array with shape [num_players, obs_size]
            Multi env (num_envs>1): numpy array with shape [num_envs, num_players, obs_size]
        """
        if not hasattr(self, '_shared_buffer_ready') or not self._shared_buffer_ready:
            raise GymError("Observations not available. Call reset() first.")
        
        if self.num_envs == 1:
            return self.shared_array[0]
        else:
            return self.shared_array

    def reset(self, league_level:int, random_seeds):
        """
        Reset the game environment to the initial state.

        Args:
            league_level: The level of the game league.
            random_seeds: Random seed(s). Can be:
                         - Single int (automatically wrapped to [random_seeds])
                         - List/array of ints (one per environment, must match num_envs)

        Returns:
            None. Access observations via the .observations property for zero-copy access.
        
        Raises:
            GymError: If league_level is invalid or seed configuration doesn't match num_envs.
        """

        # Convert single seed to array for unified handling
        if isinstance(random_seeds, int):
            random_seeds = [random_seeds]
        
        if len(random_seeds) != self.num_envs:
            raise GymError(f'The number of seeds ({len(random_seeds)}) must match num_envs ({self.num_envs}).')
            
        try:
            result = self.env.reset(league_level, random_seeds)
        except Exception as e:
            raise GymError(str(e)) from None
        
        # Setup shared buffer on first reset
        if not hasattr(self, '_shared_buffer_ready') or not self._shared_buffer_ready:
            self._setup_shared_buffers()
        
        # Set flag to indicate reset has been called
        self._reset_called = True
    
    def reset_subset(self, indices, league_level:int, random_seeds):
        """
        Reset only a subset of game environments.

        This method allows resetting only specific environments while leaving others unchanged.
        Only available in multi-environment mode (num_envs > 1).

        WARNING: Must not be called concurrently with step(). Ensure sequential calls.

        Args:
            indices: List/array of environment indices to reset (0-based, must be valid indices < num_envs).
            league_level: The level of the game league for all reset environments.
            random_seeds: List/array of random seeds (one per index, must match length of indices).

        Returns:
            None. Access observations via the .observations property for zero-copy access.
        
        Raises:
            GymError: If not in multi-env mode, indices are invalid, league_level is invalid, 
                     or seed count doesn't match indices count.
        
        Example:
            >>> env = gym.make('backtrack', num_envs=10)
            >>> env.reset(league_level=3, random_seeds=range(10))
            >>> # Reset only environments 3, 5, 7
            >>> env.reset_subset(indices=[3, 5, 7], league_level=3, random_seeds=[100, 200, 300])
        """
        if self.num_envs == 1:
            raise GymError('reset_subset() is only available for multi-environment mode (num_envs > 1).')

        # Convert to lists if needed
        if not isinstance(indices, list):
            indices = list(indices)
        if not isinstance(random_seeds, list):
            random_seeds = list(random_seeds)
        
        if len(random_seeds) != len(indices):
            raise GymError(f'The number of seeds ({len(random_seeds)}) must match number of indices ({len(indices)}).')
            
        try:
            result = self.env.resetSubset(indices, league_level, random_seeds)
        except Exception as e:
            raise GymError(str(e)) from None
        
        # Setup shared buffer on first reset (if not already done)
        if not hasattr(self, '_shared_buffer_ready') or not self._shared_buffer_ready:
            self._setup_shared_buffers()
        
        # Set flag to indicate reset has been called
        self._reset_called = True
    
    def step(self, actions):
        """
        Take a step in the game environment.

        Args:
            actions: Single env: List[List[str]] (player → actions), wrapped to multi-env format internally
                    Multi env: List[List[List[str]]] (env → player → actions)

        Returns:
            Single env: done (bool). Access observations via .observations property.
            Multi env: dones (list of bool). Access observations via .observations property.
        """

        if not self._reset_called:
            raise Exception('You must call reset before calling step for the first time.')
        
        # Wrap single-env actions to multi-env format
        if self.num_envs == 1 and len(actions) > 0 and not isinstance(actions[0][0], list):
            # actions is List[List[str]], wrap to List[List[List[str]]]
            actions = [actions]
        
        # Convert to Java format
        if isinstance(actions, list):
            actions_java = convert_to_java_list_of_lists_of_lists(actions)
        else:
            actions_java = actions
        
        result = self.env.step(actions_java)
        
        self.turns = result.turns  # Per-environment turn counters
        
        if self.num_envs == 1:
            # Single env: return only done flag
            return result.dones[0]
        else:
            # Multi env: return only dones array
            return result.dones
    
    def get_scores(self):
        """
        Get the scores of each player.

        Returns:
            Single env: List of scores for each player
            Multi env: List of lists of scores for each environment and player
        """
        scores = self.env.get_scores()
        if self.num_envs == 1:
            # Unwrap from multi-env format
            return scores[0]
        return scores
    
    def get_frame_strings(self):
        """
        Get string observations for the current frame (most recent reset or step).
        Only available when debug mode is enabled.
        
        Returns:
            Single env (num_envs=1): List[List[str]] - per player string observations
            Multi env (num_envs>1): List[List[List[str]]] - per env, per player string observations
            Empty lists if debug mode is disabled.
        
        Example:
            >>> env = gym.make('backtrack', debug=True)
            >>> env.reset(league_level=3, random_seed=12345)
            >>> strings = env.get_frame_strings()
            >>> # strings[player_idx] is a list of lines for that player
            >>> # Parse to compare with numeric observations
        """
        if not hasattr(self, 'debug_mode') or not self.debug_mode:
            if self.num_envs == 1:
                return []
            else:
                return [[] for _ in range(self.num_envs)]
        
        frame_strings = self.env.getCurrentFrameStrings()
        
        if self.num_envs == 1:
            # Unwrap from multi-env format
            return frame_strings[0] if frame_strings else []
        return frame_strings
    
    def get_timing_stats(self):
        """
        Get detailed timing statistics for performance analysis.
        Only available in multi-environment mode.
        
        Returns:
            dict: Timing breakdown or None if not available
        """
        if not hasattr(self, '_timing_stats'):
            return None
        return self._timing_stats.copy()
    
    def print_timing_stats(self):
        """
        Print detailed timing statistics in a readable format.
        Only available in multi-environment mode.
        """
        if not hasattr(self, '_timing_stats'):
            print("No timing statistics available. Call step() first.")
            return
        
        stats = self._timing_stats
        total = stats['conversion_time'] + stats['java_call_time'] + stats['numpy_conversion_time']
        count = stats['step_count']
        
        print("\n" + "="*60)
        print("Performance Breakdown (Python-Java Bridge)")
        print("="*60)
        print(f"Total steps: {count}")
        print(f"\nTotal time: {total:.3f}s")
        print(f"  Python→Java conversion: {stats['conversion_time']:.3f}s ({100*stats['conversion_time']/total:.1f}%)")
        print(f"  Java execution:         {stats['java_call_time']:.3f}s ({100*stats['java_call_time']/total:.1f}%)")
        print(f"  Java→NumPy conversion:  {stats['numpy_conversion_time']:.3f}s ({100*stats['numpy_conversion_time']/total:.1f}%)")
        print(f"\nPer-step averages:")
        print(f"  Python→Java conversion: {1000*stats['conversion_time']/count:.2f}ms")
        print(f"  Java execution:         {1000*stats['java_call_time']/count:.2f}ms")
        print(f"  Java→NumPy conversion:  {1000*stats['numpy_conversion_time']/count:.2f}ms")
        print(f"  Total per step:         {1000*total/count:.2f}ms")
        print("="*60 + "\n")
    
    def shutdown(self):
        """
        Shutdown the JVM to allow the Python process to exit cleanly.
        Call this when you're done using the environment.
        """
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.shutdown()
            except:
                pass
        
        if jpype.isJVMStarted():
            try:
                jpype.shutdownJVM()
            except:
                pass