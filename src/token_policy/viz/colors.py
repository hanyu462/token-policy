"""Named colours for drawing, as (r, g, b, a) in [0, 1].

    from token_policy.viz import colors
    c.sphere(p, 0.02, colors.RED)
"""
RED = (1.0, 0.1, 0.1, 1.0)
GREEN = (0.1, 0.9, 0.1, 1.0)
BLUE = (0.1, 0.4, 1.0, 1.0)
YELLOW = (1.0, 0.9, 0.1, 1.0)
ORANGE = (1.0, 0.5, 0.0, 1.0)
CYAN = (0.0, 0.8, 0.8, 1.0)
MAGENTA = (1.0, 0.0, 1.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
GRAY = (0.6, 0.6, 0.6, 0.8)
BLACK = (0.0, 0.0, 0.0, 1.0)
TRANSPARENT = (0.0, 0.0, 0.0, 0.0)


def with_alpha(color, alpha: float):
    """Same colour with a different opacity."""
    return (color[0], color[1], color[2], float(alpha))
