import jax
from jax import numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

from envs.frozen_lake import FrozenLake


class PolicyNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.num_actions)(x)
        return x


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['obs'])
        log_probs = jax.nn.log_softmax(logits)
        log_prob_actions = jnp.take_along_axis(
            log_probs,
            batch['actions'][..., None],
            axis=1
        ).squeeze()
        loss = -log_prob_actions * batch['rewards']
        return loss.mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main():
    key = jax.random.key(0)

    env = FrozenLake()
    env_params = env.default_params.replace(is_slippery=False)

    num_states = env.observation_space(env_params).n
    num_actions = env.num_actions

    key, agent_key = jax.random.split(key, 2)

    agent = PolicyNetwork(num_actions)
    params = agent.init(agent_key, jnp.zeros((1, num_states,)))['params']
    tx = optax.adam(learning_rate=0.0001)

    agent_state = train_state.TrainState.create(
        apply_fn=agent.apply,
        params=params,
        tx=tx
    )

    gamma = 0.99
    total_reward = 0
    total_loss = 0
    for i in range(100000):
        key, key_reset = jax.random.split(key, 2)
        obs, state = env.reset(key_reset, env_params)

        episode = []

        for _ in range(1000):
            key, action_selection_key = jax.random.split(key, 2)

            one_hot_obs = jax.nn.one_hot(obs, num_states)[None, ...]
            logits = agent_state.apply_fn({'params': agent_state.params}, one_hot_obs)
            action = int(jax.random.categorical(action_selection_key, logits, axis=1)[0])

            key, key_step = jax.random.split(key, 2)
            n_obs, state, reward, done, _ = env.step(key_step, state, action, env_params)
            total_reward += reward

            episode.append({
                'obs': jax.nn.one_hot(obs, num_states),
                'action': action,
                'reward': reward,
            })

            if done:
                break
            obs = n_obs
        
        rewards = [step['reward'] for step in episode]
        returns = []

        discounted_return = 0.0
        for r in reversed(rewards):
            discounted_return = r + gamma * discounted_return
            returns.insert(0, discounted_return)

        returns = jnp.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        batch_obs = jnp.array([step['obs'] for step in episode])
        batch_actions = jnp.array([step['action'] for step in episode])

        batch = {
            'obs': batch_obs,
            'actions': batch_actions,
            'rewards': returns,
        }

        agent_state, loss = train_step(agent_state, batch)
        total_loss += loss
    
        if i % 50 == 0:
            print(f'Episode: {i}, Avg Reward: {total_reward / 50:.2e}, Avg Loss: {total_loss / 50:.4e}')
            total_reward = 0
            total_loss = 0


if __name__ == '__main__':
    main()