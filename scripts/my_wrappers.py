# Jacques' re-implementation of wrappers for JAX environments, following purejaxrl/wrappers.py.

from functools import partial
from typing import Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import navix as nx
import numpy as np
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from flax import struct
from gymnax.environments import environment, spaces


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env: environment.Environment):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name: str):
        return getattr(self._env, name)


class FlattenObservation(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params: environment.EnvParams) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))  # flatten observation
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: chex.Array,
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))  # flatten observation
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: chex.Array,
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class BraxGymnaxWrapper(environment.Environment):
    def __init__(self, env_name: str, backend: str = "positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(
        self, key: chex.PRNGKey, params=None
    ) -> Tuple[chex.Array, environment.EnvState]:
        state = self._env.reset(rng=key)
        return state.obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: chex.Array,
        params=None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class NavixGymnaxWrapper(environment.Environment):
    def __init__(self, env_name: str):
        self._env: nx.Environment = nx.make(env_name)

    def reset(self, key: chex.PRNGKey, params=None):
        timestep = self._env.reset(key)
        return timestep.observation, timestep

    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: chex.Array,
        params=None,
    ):
        timestep = self._env.step(state, action)
        return timestep.observation, timestep, timestep.reward, timestep.is_done(), {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(
            num_categories=self._env.action_space.maximum.item() + 1,
        )


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state: environment.EnvState, action: chex.Array, params=None):
        clipped_action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, clipped_action, params)


class TransformObservation(GymnaxWrapper):
    def __init__(
        self,
        env: environment.Environment,
        transform_obs: Callable[[chex.Array], chex.Array],
    ):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key: chex.PRNGKey, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: chex.Array,
        params=None,
    ):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    def __init__(
        self, env: environment.Environment, transform_reward: Callable[[float], float]
    ):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: chex.Array,
        params=None,
    ):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: chex.Array
    var: chex.Array
    count: float
    env_state: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env: VecEnv):
        super().__init__(env)

    def reset(
        self, key: chex.PRNGKey, params=None
    ) -> Tuple[chex.Array, NormalizeVecObsEnvState]:
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / (jnp.sqrt(state.var) + 1e-8), state

    def step(
        self,
        key: chex.PRNGKey,
        state: NormalizeVecObsEnvState,
        action: chex.Array,
        params=None,
    ) -> Tuple[chex.Array, NormalizeVecObsEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / (jnp.sqrt(state.var) + 1e-8),
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: float
    var: float
    count: float
    return_val: chex.Array
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env: VecEnv, gamma: float):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key: chex.PRNGKey, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: NormalizeVecRewEnvState,
        action: chex.Array,
        params=None,
    ) -> Tuple[chex.Array, NormalizeVecRewEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / (jnp.sqrt(state.var) + 1e-8), done, info


if __name__ == "__main__":
    # Example usage

    from typing import NamedTuple

    class Transition(NamedTuple):
        done: chex.Array
        action: chex.Array
        reward: chex.Array
        obs: chex.Array
        info: chex.Array

    def make_example_usage(config):

        env, env_params = BraxGymnaxWrapper(env_name=config["ENV_NAME"]), None
        action_space = env.action_space(env_params)
        env = LogWrapper(env)
        env = ClipAction(env)
        env = VecEnv(env)
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, gamma=config["GAMMA"])

        def example_usage(rng):

            rng, _rng = jax.random.split(rng)

            # You would initialize the network, optimizer and train state here
            train_state = None

            # Instead, we'll just sample random actions from the action space
            def sample_single_action(key):
                return action_space.sample(key)

            sample_action = jax.vmap(sample_single_action)

            rng, _rng = jax.random.split(rng)

            rng_reset = jax.random.split(_rng, config["NUM_ENVS"])

            obs, env_state = env.reset(rng_reset, env_params)

            jax.debug.print("Initial Observation: {}", obs)
            jax.debug.print("---")

            def _env_step(runner_state, _):

                train_state, env_state, last_obs, rng = runner_state

                rng, _rng = jax.random.split(rng)
                # rng_sample unneeded if using a policy network (just use _rng)
                rng_sample = jax.random.split(_rng, config["NUM_ENVS"])
                action = sample_action(rng_sample)

                jax.debug.print("Action: {}", action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obs, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )

                jax.debug.print("Observation: {}", obs)
                jax.debug.print("Reward: {}", reward)
                jax.debug.print("Done: {}", done)
                jax.debug.print("Info: {}", info)

                transition = Transition(
                    done=done, action=action, reward=reward, obs=obs, info=info
                )
                runner_state = (train_state, env_state, obs, rng)

                jax.debug.print("---")

                return runner_state, transition

            rng, _rng = jax.random.split(rng)
            runner_state = (train_state, env_state, obs, _rng)

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, length=config["NUM_STEPS"]
            )

            jax.debug.print("Trajectory Batch: {}", traj_batch)

        return example_usage

    config = {
        "NUM_ENVS": 4,
        "NUM_STEPS": 5,
        "ENV_NAME": "hopper",
        "GAMMA": 0.99,
    }
    rng = jax.random.PRNGKey(0)
    example_usage_jit = jax.jit(make_example_usage(config))
    example_usage_jit(rng)
