import hashlib
import gym
from gym.envs.registration import register

def make(
        domain_name,
        task_name,
        task_params,
        seed=1,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        episode_length=1000,
        environment_kwargs=None,
        time_limit=None,
        channels_first=True
):
    unique_args = int(hashlib.sha1(str(locals()).encode('utf8')).hexdigest(), 16) % (10 ** 8)

    env_id = f'dmc_{domain_name}_{task_name}_s{seed}_{unique_args}-v1'

    if from_pixels:
        assert not visualize_reward, 'cannot use visualize reward when learning from pixels'

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if not env_id in gym.envs.registry.env_specs:
        task_kwargs = {}
        if seed is not None:
            task_kwargs['random'] = seed
        if time_limit is not None:
            task_kwargs['time_limit'] = time_limit
        task_kwargs['params'] = task_params
        register(
            id=env_id,
            entry_point='dmc3gym.wrappers:DMCWrapper',
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)
