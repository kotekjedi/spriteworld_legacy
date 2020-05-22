  
# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3
"""Generators for producing lists of sprites based on factor distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import torch
from spriteworld import sprite
from spriteworld import factor_distributions as distribs


def generate_sprites(factor_dist, num_sprites=1):
  """Create callable that samples sprites from a factor distribution.
  Args:
    factor_dist: The factor distribution from which to sample. Should be an
      instance of factor_distributions.AbstractDistribution.
    num_sprites: Int or callable returning int. Number of sprites to generate
      per call.
  Returns:
    _generate: Callable that returns a list of Sprites.
  """

  def _generate():
    n = num_sprites() if callable(num_sprites) else num_sprites

    sprites = []
    current = []
    for _ in range(n):
        temp_sprite = sprite.Sprite(**factor_dist.sample())
        while temp_sprite.shape in current:
          temp_sprite = sprite.Sprite(**factor_dist.sample())

        current.append(temp_sprite.shape)

        if temp_sprite.shape == 'square':
            x1 =10.23245679643229
            x2 =3.0928442981097133
            y1 =20.495181258545415
            y2 =21.281283533153676
            scale1 =19.089783913500117
            scale2 =2.9445437594982042
            col1 =16.08196323063433
            col2 =2.1070937529079767
            factors = distribs.Product([
            distribs.Beta('x', x1, x2),
            distribs.Beta('y', y1, y2),
            distribs.Discrete('shape', [temp_sprite.shape]),
            distribs.Beta('scale', scale1,scale2),
            # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
            distribs.Beta('c0', col1, col2),
            distribs.Continuous('c1', 1., 1.),
            distribs.Continuous('c2', 1., 1.),
            ])

        elif temp_sprite.shape == 'triangle':
            x1 =4.834302546465699
            x2 =9.109414385787378
            y1 =6.734115828769332
            y2 =10.623775195489053
            scale1 =17.702226349877975
            scale2 =22.955766633205197
            col1 =11.815705623267306
            col2 =23.2806752586176

            factors = distribs.Product([
            distribs.Beta('x', x1, x2),
            distribs.Beta('y', y1, y2),
            distribs.Discrete('shape', [temp_sprite.shape]),
            distribs.Beta('scale', scale1,scale2),
            # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
            distribs.Beta('c0', col1, col2),
            distribs.Continuous('c1', 1., 1.),
            distribs.Continuous('c2', 1., 1.),
            ])

        elif temp_sprite.shape == 'pentagon':
            x1 =2.5283306085768826
            x2 =13.37857404643091
            y1 =15.662147422996169
            y2 =13.676905991754056
            scale1 =17.565014615262086
            scale2 =13.508529849526141
            col1 =11.934008307559518
            col2 =10.2405234460427
            factors = distribs.Product([
            distribs.Beta('x', x1, x2),
            distribs.Beta('y', y1, y2),
            distribs.Discrete('shape', [temp_sprite.shape]),
            distribs.Beta('scale', scale1,scale2),
            # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
            distribs.Beta('c0', col1, col2),
            distribs.Continuous('c1', 1., 1.),
            distribs.Continuous('c2', 1., 1.),
            ])

        elif temp_sprite.shape == 'hexagon':
            x1 =22.39339816543392
            x2 =21.268933430450854
            y1 =15.4426245650715
            y2 =23.541179853521466
            scale1 =8.482968396689424
            scale2 =21.976821208427644
            col1 =5.267669658723397
            col2 =20.600361827332314
            factors = distribs.Product([
            distribs.Beta('x', x1, x2),
            distribs.Beta('y', y1, y2),
            distribs.Discrete('shape', [temp_sprite.shape]),
            distribs.Beta('scale', scale1,scale2),
            # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
            distribs.Beta('c0', col1, col2),
            distribs.Continuous('c1', 1., 1.),
            distribs.Continuous('c2', 1., 1.),
            ])

        elif temp_sprite.shape == 'spoke_4':
            x1 =13.456411110280614
            x2 =22.391767509044413
            y1 =4.111114984741711
            y2 =8.832948313668181
            scale1 =13.763110818048586
            scale2 =2.51798500580912
            col1 =9.487345285873175
            col2 =14.85934849805914
            factors = distribs.Product([
            distribs.Beta('x', x1, x2),
            distribs.Beta('y', y1, y2),
            distribs.Discrete('shape', [temp_sprite.shape]),
            distribs.Beta('scale', scale1,scale2),
            # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
            distribs.Beta('c0', col1, col2),
            distribs.Continuous('c1', 1., 1.),
            distribs.Continuous('c2', 1., 1.),
            ])

        elif temp_sprite.shape == 'spoke_5':

            x1 =6.317495984173592
            x2 =14.42695863585005
            y1 =14.392050785533002
            y2 =23.794986731719288
            scale1 =12.357798701299764
            scale2 =17.152799524019557
            col1 =9.327202992582688
            col2 =9.226934147141376
            factors = distribs.Product([
            distribs.Beta('x', x1, x2),
            distribs.Beta('y', y1, y2),
            distribs.Discrete('shape', [temp_sprite.shape]),
            distribs.Beta('scale', scale1,scale2),
            # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
            distribs.Beta('c0', col1, col2),
            distribs.Continuous('c1', 1., 1.),
            distribs.Continuous('c2', 1., 1.),
            ])

        elif temp_sprite.shape == 'spoke_6':
            x1 =15.41214880790467
            x2 =18.003492016871608
            y1 =17.44700259100445
            y2 =16.948723697275597
            scale1 =22.34867970899941
            scale2 =17.613695936551235
            col1 =12.857871878835631
            col2 =9.410023222672462
            factors = distribs.Product([
            distribs.Beta('x', x1, x2),
            distribs.Beta('y', y1, y2),
            distribs.Discrete('shape', [temp_sprite.shape]),
            distribs.Beta('scale', scale1,scale2),
            # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
            distribs.Beta('c0', col1, col2),
            distribs.Continuous('c1', 1., 1.),
            distribs.Continuous('c2', 1., 1.),
            ])

        elif temp_sprite.shape == 'star_4':

            x1 =6.794448885245786
            x2 =12.364891466499806
            y1 =19.994745864858775
            y2 =22.82343780355094
            scale1 =9.061912292390144
            scale2 =11.375690804235813
            col1 =10.088677412554267
            col2 =18.624525742581024
            factors = distribs.Product([
            distribs.Beta('x', x1, x2),
            distribs.Beta('y', y1, y2),
            distribs.Discrete('shape', [temp_sprite.shape]),
            distribs.Beta('scale', scale1,scale2),
            # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
            distribs.Beta('c0', col1, col2),
            distribs.Continuous('c1', 1., 1.),
            distribs.Continuous('c2', 1., 1.),
            ])

        elif temp_sprite.shape == 'star_5':

            x1 =21.963199992980194
            x2 =24.38383725086071
            y1 =24.964822957405055
            y2 =16.398682135890102
            scale1 =4.223489127851181
            scale2 =13.478902714779057
            col1 =22.59618525644741
            col2 =10.012697617533178
            factors = distribs.Product([
            distribs.Beta('x', x1, x2),
            distribs.Beta('y', y1, y2),
            distribs.Discrete('shape', [temp_sprite.shape]),
            distribs.Beta('scale', scale1,scale2),
            # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
            distribs.Beta('c0', col1, col2),
            distribs.Continuous('c1', 1., 1.),
            distribs.Continuous('c2', 1., 1.),
            ])

        elif temp_sprite.shape == 'star_6':

            x1=16.929802261362223
            x2=11.581933601608984
            y1 =23.171125447416305
            y2 =15.368603378028528
            scale1 =17.51702940194395
            scale2 =10.43361003805039
            col1 =10.98218742901024
            col2 =13.678590535311319
            factors = distribs.Product([
            distribs.Beta('x', x1, x2),
            distribs.Beta('y', y1, y2),
            distribs.Discrete('shape', [temp_sprite.shape]),
            distribs.Beta('scale', scale1,scale2),
            # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
            distribs.Beta('c0', col1, col2),
            distribs.Continuous('c1', 1., 1.),
            distribs.Continuous('c2', 1., 1.),
            ])


        sprites.append(sprite.Sprite(**factors.sample()))


    return sprites

  return _generate


def chain_generators(*sprite_generators):
  """Chain generators by concatenating output sprite sequences.
  Essentially an 'AND' operation over sprite generators. This is useful when one
  wants to control the number of samples from the modes of a multimodal sprite
  distribution.
  Note that factor_distributions.Mixture provides weighted mixture
  distributions, so chain_generators() is typically only used when one wants to
  forces the different modes to each have a non-zero number of sprites.
  Args:
    *sprite_generators: Callable sprite generators.
  Returns:
    _generate: Callable returning a list of sprites.
  """

  def _generate():
    return list(
        itertools.chain(*[generator() for generator in sprite_generators]))

  return _generate


def sample_generator(sprite_generators, p=None):
  """Sample one element from a set of sprite generators.
  Essential an 'OR' operation over sprite generators. This returns a callable
  that samples a generator from sprite_generators and calls it.
  Note that if sprite_generators each return 1 sprite, this functionality can be
  achieved with factor_distributions.Mixture, so sample_generator is typically
  used when sprite_generators each return multiple sprites. Effectively it
  allows dependant sampling from a multimodal factor distribution.
  Args:
    sprite_generators: Iterable of callable sprite generators.
    p: Probabilities associated with each generator. If None, assumes uniform
      distribution.
  Returns:
    _generate: Callable sprite generator.
  """

  def _generate():
    sample_index = np.random.choice(len(sprite_generators), p=p)
    sampled_generator = sprite_generators[sample_index]
    return sampled_generator()

  return _generate


def shuffle(sprite_generator):
  """Randomize the order of sprites sample from sprite_generator.
  This is useful because sprites are z-layered with occlusion according to their
  order, so is sprite_generator is the output of chain_generators(), then
  sprites from some component distributions will always be behind sprites from
  others.
  An alternate design would be to let the environment handle sprite ordering,
  but this design is preferable because the order can be controlled more finely.
  For example, this allows the user to specify one sprite (e.g. the agent's
  body) to always be in the foreground while all the others are randomly
  ordered.
  Args:
    sprite_generator: Callable return a list of sprites.
  Returns:
    _generate: Callable sprite generator.
  """

  def _generate():
    sprites = sprite_generator()
    order = np.arange(len(sprites))
    np.random.shuffle(order)
    return [sprites[i] for i in order]

  return _generate