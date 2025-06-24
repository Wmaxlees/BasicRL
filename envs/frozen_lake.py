from gymnax.environments import environment, spaces
from flax import struct
import jax.numpy as jnp
import jax


@struct.dataclass
class EnvState(environment.EnvState):
    player_pos: int
    time: int

@struct.dataclass
class EnvParams(environment.EnvParams):
    is_slippery: bool = False
    max_steps_in_episode: int = 100

# Using integers for tile types: 0: Start, 1: Frozen, 2: Hole, 3: Goal
FROZEN_LAKE_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

def string_map_to_int_map(string_map):
    """Converts the string map to a numerical grid and gets tile locations."""
    char_to_int = {'S': 0, 'F': 1, 'H': 2, 'G': 3}
    int_map = jnp.array([[char_to_int[c] for c in row] for row in string_map])
    return int_map

class FrozenLake(environment.Environment[EnvState, EnvParams]):
    """JAX implementation of FrozenLake-v1 environment."""

    def __init__(self, map_name: str = "4x4"):
        super().__init__()
        # For now, we'll stick to the classic 4x4 map
        self.map_shape = (4, 4)
        self.int_map = string_map_to_int_map(FROZEN_LAKE_MAP).flatten()

        # Actions: 0: Left, 1: Down, 2: Right, 3: Up
        self.action_to_delta = jnp.array([
            [0, -1], # Left
            [1, 0],  # Down
            [0, 1],  # Right
            [-1, 0]  # Up
        ])

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]:
        """Perform single timestep state transition."""
        
        # --- Handle Slipperiness ---
        def slippery_action(key, a):
            # With probability 1/3, the chosen action is taken.
            # Otherwise, one of the perpendicular actions is taken with 1/3 prob each.
            p = jnp.array([0.0, 0.0, 0.0, 0.0]).at[a].set(1/3)
            p = p.at[(a - 1) % 4].add(1/3) # Perpendicular action 1
            p = p.at[(a + 1) % 4].add(1/3) # Perpendicular action 2
            return jax.random.choice(key, 4, p=p)

        def non_slippery_action(key, a):
            return a
        
        action = jax.lax.cond(
            params.is_slippery,
            slippery_action,
            non_slippery_action,
            key,
            action
        )
        
        # --- Calculate new position ---
        y, x = self.to_yx(state.player_pos)
        delta = self.action_to_delta[action]
        new_y, new_x = jnp.clip(y + delta[0], 0, self.map_shape[0] - 1), \
                       jnp.clip(x + delta[1], 0, self.map_shape[1] - 1)
        
        new_pos = self.to_pos(new_y, new_x)
        
        # --- Check for termination and calculate reward ---
        tile_type = self.int_map[new_pos]
        
        # Reward is 1 only if the agent reaches the goal
        reward = (tile_type == 3).astype(jnp.float32)
        
        # Episode ends if agent reaches goal or falls in a hole
        done = jnp.logical_or(tile_type == 2, tile_type == 3)
        
        # Update state
        state = EnvState(player_pos=new_pos, time=state.time + 1)
        done = jnp.logical_or(done, state.time >= params.max_steps_in_episode)
        
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state."""
        # Agent always starts at 'S', which is position 0
        state = EnvState(player_pos=0, time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return agent's position as observation."""
        return state.player_pos

    def to_yx(self, pos: int):
        """Convert position index to (y, x) coordinates."""
        return pos // self.map_shape[1], pos % self.map_shape[1]

    def to_pos(self, y: int, x: int):
        """Convert (y, x) coordinates to position index."""
        return y * self.map_shape[1] + x
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        tile_type = self.int_map[state.player_pos]
        # Episode ends if agent is on a Hole (2) or Goal (3) tile
        done_tile = jnp.logical_or(tile_type == 2, tile_type == 3)
        # Episode also ends if time exceeds max steps
        done_time = state.time >= params.max_steps_in_episode
        return jnp.logical_or(done_tile, done_time)
        
    @property
    def name(self) -> str: return "FrozenLake-v0"
    @property
    def num_actions(self) -> int: return 4
    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete: return spaces.Discrete(4)
    def observation_space(self, params: EnvParams) -> spaces.Discrete: return spaces.Discrete(self.map_shape[0] * self.map_shape[1])
