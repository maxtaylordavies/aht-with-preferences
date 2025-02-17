"""
2D rendering of the level based foraging domain
"""

import math
import os
import sys

import numpy as np
import math
import six
from gymnasium import error

from src.utils import from_one_hot
from .lbf import LBFEnvState

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"

import pyglet
from pyglet import shapes
from pyglet.gl import *


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 100
        self.icon_size = 40

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)
        pyglet.resource.path = [os.path.join(script_dir, "icons")]
        pyglet.resource.reindex()

        self.agent_icon = pyglet.resource.image("agent.png")
        self.food_icons = [
            pyglet.resource.image(fn) for fn in ["apple.png", "orange.png", "pear.png"]
        ]

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env_state: LBFEnvState, return_rgb_array=False):
        glClearColor(*_WHITE, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_food(env_state)
        self._draw_players(env_state)

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        v_lines, h_lines = [], []

        for r in range(self.rows + 1):
            v_lines.append(shapes.Line(
                0,  # LEFT X
                (self.grid_size + 1) * r + 1,  # Y
                (self.grid_size + 1) * self.cols,  # RIGHT X
                (self.grid_size + 1) * r + 1,  # Y,
                thickness=2,
                color=_BLACK,
                batch=batch,
            ))

        for c in range(self.cols + 1):
            h_lines.append(shapes.Line(
                (self.grid_size + 1) * c + 1,  # X
                0,  # BOTTOM Y
                (self.grid_size + 1) * c + 1,  # X
                (self.grid_size + 1) * self.rows,  # TOP X
                thickness=2,
                color=_BLACK,
                batch=batch,
            ))

        batch.draw()

    def _draw_food(self, env_state: LBFEnvState):
        foods, batch = [], pyglet.graphics.Batch()

        for i in range(len(env_state.fruit_locs)):
            if env_state.fruit_consumed[i]:
                continue
            row, col = env_state.fruit_locs[i]
            food_type = env_state.fruit_types[i]
            foods.append(
                pyglet.sprite.Sprite(
                    self.food_icons[food_type],
                    (self.grid_size + 1) * col,
                    self.height - (self.grid_size + 1) * (row + 1),
                    batch=batch,
                )
            )
        for f in foods:
            f.update(scale=self.grid_size / f.width)
        batch.draw()

        for i in range(len(env_state.fruit_locs)):
            if env_state.fruit_consumed[i]:
                continue
            row, col = env_state.fruit_locs[i]
            self._draw_badge(row, col, env_state.fruit_levels[i])

    def _draw_players(self, env_state: LBFEnvState):
        players, batch = [], pyglet.graphics.Batch()

        for i in range(len(env_state.agent_locs)):
            row, col = env_state.agent_locs[i]
            players.append(
                pyglet.sprite.Sprite(
                    self.agent_icon,
                    (self.grid_size + 1) * col,
                    self.height - (self.grid_size + 1) * (row + 1),
                    batch=batch,
                )
            )

        for p in players:
            p.update(scale=self.grid_size / p.width)
        batch.draw()

        for i in range(len(env_state.agent_locs)):
            row, col = env_state.agent_locs[i]
            if i == 0:
                self._draw_player_marker(row, col)
            self._draw_badge(row, col, env_state.agent_levels[i])
            self._draw_player_feature(row, col, env_state.agent_types[i])

    def _draw_player_feature(self, row, col, agent_type):
        bar_x = (col * (self.grid_size + 1)) + 1
        bar_y = self.height - (self.grid_size + 1) * (row + 1)
        bar_length, bar_height = self.grid_size, self.grid_size / 8
        bar_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][agent_type % 3]
        rectangle = pyglet.shapes.Rectangle(
            bar_x, bar_y, bar_length, bar_height, color=bar_color, batch=None
        )
        rectangle.draw()

    def _draw_player_marker(self, row, col):
        # draw an upside-down triangle above the player
        marker_x = col * (self.grid_size + 1) + (self.grid_size + 1) / 2
        marker_y = self.height - (self.grid_size + 1) * (row + 0.1)
        size = self.grid_size / 5
        triangle = shapes.Triangle(
            marker_x, marker_y, marker_x - size, marker_y + size, marker_x + size, marker_y + size, color=_RED
        )
        triangle.draw()

    def _draw_badge(self, row, col, level):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (1 / 4) * (self.grid_size + 1)
        )

        circle = shapes.Circle(
            badge_x, badge_y, radius, segments=resolution, color=_WHITE
        )
        circle.draw()

        label = pyglet.text.Label(
            str(level),
            font_name="Times New Roman",
            font_size=self.icon_size / 2,
            weight="bold",
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 255),
        )
        label.draw()
