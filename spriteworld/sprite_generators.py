  
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
    for _ in range(n):
        temp_sprite = sprite.Sprite(**factor_dist.sample())

        if temp_sprite.shape == 'square':
            x1= 2.197230602346917
            x2=1.153724437287404
            y1=0.14632802055043548
            y2 =3.09618161775054
            scale1 =3.9938863588625617
            scale2 =3.430671410516569
            col1 =0.3031965373635036
            col2= 2.004732703244329
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
            x1= 3.2337187044256734
            x2= 4.46846785338019
            y1= 1.5764973135915938
            y2= 1.9024976963413818
            scale1= 3.304188844571709
            scale2= 1.5666843281416285
            col1= 2.811260885631706
            col2= 1.8571023349642906
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
            x1= 2.316681683653071
            x2= 0.9451377924133489
            y1= 0.12308322529892039
            y2= 1.6221723682425204
            scale1= 1.0137295905697148
            scale2= 2.314717341616334
            col1= 0.5085497273092809
            col2= 4.921854794848134
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
            x1= 3.1047701775435685
            x2= 0.1980905512508805
            y1= 2.600868402830089
            y2= 3.6497584357516937
            scale1= 2.276033013290801
            scale2= 3.32255393221239
            col1= 0.48163306224892377
            col2= 3.6558056556007683
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
            x1= 1.9840903244426307
            x2= 1.7762997905178244
            y1= 3.55189596219218
            y2= 4.797048691110845
            scale1= 1.4750382781346896
            scale2= 3.5852513722287553
            col1= 3.648923677539326
            col2= 2.1215537199740098
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

            x1= 1.7681831718253354
            x2= 4.340274802094907
            y1= 4.249931099155908
            y2= 4.885983116606701
            scale1= 4.989081642470982
            scale2= 1.6468501626401895
            col1= 1.4491490932086
            col2= 2.501299141248418
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

            x1= 2.220316484476325
            x2= 3.1638895303785266
            y1= 4.69949080333609
            y2= 2.996728404291259
            scale1= 3.346740598620086
            scale2= 3.701927184129711
            col1= 0.23116290833912917
            col2= 3.5679113688087485
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

            x1= 4.289398237605896
            x2= 3.950645432386338
            y1= 0.7395265643456012
            y2= 2.319350624441188
            scale1= 3.0923223891784968
            scale2= 3.6853577401072393
            col1= 2.1822301466022207
            col2= 3.3880528807330594
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

            x1= 2.222425872958354
            x2= 3.015199240931054
            y1= 4.6046549586501175
            y2= 2.4352230675934536
            scale1= 3.1053136410610884
            scale2= 0.8431738953685974
            col1= 3.4227613617687753
            col2= 4.9841699242574595
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

            x1= 1.8547333505322596
            x2= 3.487907763354374
            y1= 4.430307176169718
            y2= 2.624024997203754
            scale1= 4.886703358159286
            scale2= 3.5595215660755186
            col1= 1.294932804380183
            col2= 0.10686931606742181
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