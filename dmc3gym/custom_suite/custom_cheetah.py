# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Cheetah Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections

from dm_control import mujoco
from dmc3gym.custom_suite import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
from lxml import etree

# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Running speed above which reward is 1.
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


def get_model_and_assets(body_length):
  """Returns a tuple containing the model XML string and a dict of assets."""
  #suite_dir = os.path.dirname(os.path.dirname(__file__))
  #xml_string = resources.GetResource(os.path.join(
  #     suite_dir, 'custom_suite/custom_cheetah.xml'))
  xml_string = common.read_model('cheetah.xml')
  mjcf = etree.fromstring(xml_string)
  body = mjcf.find('./worldbody/body/')
  for i in range(6):
    body = body.getnext()
  body.set('fromto', "-%g 0 0 %g 0 0"%(body_length, body_length))
  head = body.getnext()
  head.set('pos', "%g 0 .1"%(body_length+0.1))
  thigh =  head.getnext()
  thigh.set('pos', "-%g 0 0"%body_length)
  thigh = thigh.getnext()
  thigh.set('pos', "%g 0 0"%body_length) 
  return etree.tostring(mjcf, pretty_print=True), common.ASSETS
  #return common.read_model('cheetah.xml'), common.ASSETS


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, params=None, environment_kwargs=None):
  """Returns the run task."""
  physics = []
  for body_length in params:
    physic = Physics.from_xml_string(*get_model_and_assets(body_length))
    physics.append(physic)
  task = Cheetah(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add('benchmarking')
def runmv(time_limit=_DEFAULT_TIME_LIMIT, random=None, params=None, environment_kwargs=None):
  """Returns the run task."""
  physics = []
  for mass in params:
    physic = Physics.from_xml_string(*get_model_and_assets(0.5))
    physic.model.body_mass[:] = mass
    physics.append(physic)
  task = Cheetah(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Cheetah domain."""

  def speed(self):
    """Returns the horizontal speed of the Cheetah."""
    return self.named.data.sensordata['torso_subtreelinvel'][0]


class Cheetah(base.Task):
  """A `Task` to train a running Cheetah."""

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    # The indexing below assumes that all joints have a single DOF.
    assert physics.model.nq == physics.model.njnt
    is_limited = physics.model.jnt_limited == 1
    lower, upper = physics.model.jnt_range[is_limited].T
    physics.data.qpos[is_limited] = self.random.uniform(lower, upper)
    
    # Stabilize the model before the actual simulation.
    for _ in range(200):
      physics.step()

    physics.data.time = 0
    self._timeout_progress = 0
    super(Cheetah, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state, ignoring horizontal position."""
    obs = collections.OrderedDict()
    # Ignores horizontal position to maintain translational invariance.
    obs['position'] = physics.data.qpos[1:].copy()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    return rewards.tolerance(physics.speed(),
                             bounds=(_RUN_SPEED, float('inf')),
                             margin=_RUN_SPEED,
                             value_at_margin=0,
                             sigmoid='linear')
