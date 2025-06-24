import jax
from jax import numpy as jnp
from typing import NamedTuple
import array
from functools import partial

from envs.frozen_lake import FrozenLake

class AgentState(NamedTuple):
    q_table: jnp.ndarray


class AgentParams(NamedTuple):
    alpha: float
    gamma: float
    epsilon: float
    num_actions: int


@partial(jax.jit, static_argnames=('agent_params'))
def choose_action(key: array.array, agent_state: AgentState, s: jnp.ndarray, agent_params: AgentParams):
    key, epsilon_key, action_key = jax.random.split(key, 3)

    q_values = agent_state.q_table[s, :]
    max_q = jnp.max(q_values)
    best_actions = (q_values == max_q)
    best_action_choice = jax.random.choice(action_key, jnp.arange(agent_params.num_actions), p=best_actions/best_actions.sum())

    action = jax.lax.select(
        jax.random.uniform(epsilon_key) < agent_params.epsilon,
        jax.random.randint(key, shape=(), minval=0, maxval=agent_params.num_actions),
        best_action_choice
    )

    return key, action


@partial(jax.jit, static_argnames=('agent_params'))
def update_q_table(agent_state: AgentState, s: jnp.ndarray, a: int, r: float, s_prime: jnp.ndarray, agent_params: AgentParams):
    new_q_table = agent_state.q_table.at[s, a].set(
        agent_state.q_table[s, a] + agent_params.alpha * (r + agent_params.gamma * jnp.max(agent_state.q_table[s_prime, :]) - agent_state.q_table[s, a])
    )

    return AgentState(new_q_table)


def main():
    key = jax.random.key(0)

    env = FrozenLake()
    env_params = env.default_params.replace(is_slippery=False)

    num_states = env.observation_space(env_params).n
    num_actions = env.num_actions

    q_table = jnp.zeros((num_states, num_actions,))
    agent_state = AgentState(q_table)

    agent_params = AgentParams(.1, .99, .1, num_actions)

    total_reward = 0
    for i in range(1000):
        key, key_reset = jax.random.split(key, 2)
        obs, state = env.reset(key_reset, env_params)

        for _ in range(1000):
            key, action = choose_action(key, agent_state, obs, agent_params)

            key, key_step = jax.random.split(key, 2)
            n_obs, state, reward, done, _ = env.step(key_step, state, action, env_params)
            total_reward += reward

            agent_state = update_q_table(agent_state, obs, action, reward, n_obs, agent_params)

            if done:
                break
            obs = n_obs
    
        if i % 50 == 0:
            print(f'Episode: {i}, Avg Reward: {total_reward / 50:.2e}')
            total_reward = 0


if __name__ == '__main__':
    main()