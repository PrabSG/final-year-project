from rlpyt.utils.collections import namedarraytuple
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])

def samples_to_buffer(samples):
  """
  Defines how to add data from sampler into the replay buffer. Called
  in optimize_agent() if samples are provided to that method.  In
  asynchronous mode, will be called in the memory_copier process.
  """
  return SamplesToBuffer(
    observation=samples.env.observation,
    action=samples.agent.action,
    reward=samples.env.reward,
    done=samples.env.done,
  )

def initialize_replay_buffer(algo, examples, batch_spec, async_=False):
  """Initializes a sequence replay buffer with single frame observations"""
  example_to_buffer = SamplesToBuffer(
    observation=examples["observation"],
    action=examples["action"],
    reward=examples["reward"],
    done=examples["done"],
  )
  replay_kwargs = dict(
    example=example_to_buffer,
    size=algo.replay_size,
    B=batch_spec.B,
    rnn_state_interval=0,  # do not save rnn state
    discount=algo.discount,
    n_step_return=algo.n_step_return,
  )
  replay_buffer = UniformSequenceReplayBuffer(**replay_kwargs)
  return replay_buffer
