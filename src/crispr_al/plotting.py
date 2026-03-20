"""Publication-quality plotnine theme and colour palette for the CRISPR-AL project.

Exports
-------
PUBLICATION_COLORS : list[str]
    Nine-colour hex palette suitable for categorical data.
theme_publication(base_size, base_family) -> plotnine.theme
    Minimal, white-background theme modelled on ggplot2's theme_bw.
scale_fill_publication(**kwargs) -> plotnine.scale
    Discrete fill scale using PUBLICATION_COLORS.
scale_color_publication(**kwargs) -> plotnine.scale
    Discrete colour scale using PUBLICATION_COLORS.
scale_colour_publication(**kwargs) -> plotnine.scale
    Alias for scale_color_publication.
"""

from plotnine import theme, element_blank, element_line, element_rect, element_text
from plotnine.scales import scale_fill_manual, scale_color_manual

PUBLICATION_COLORS = [
    "#2166AC",  # blue
    "#D6604D",  # red
    "#4DAC26",  # green
    "#8073AC",  # purple
    "#F4A582",  # salmon
    "#92C5DE",  # light blue
    "#A6D854",  # lime
    "#FD8D3C",  # orange
    "#878787",  # grey
]


def theme_publication(base_size: int = 12, base_family: str = "sans-serif"):
    """Return a minimal publication-ready plotnine theme.

    Parameters
    ----------
    base_size : int
        Base font size in points.
    base_family : str
        Font family for all text elements.
    """
    return theme(
        panel_background=element_rect(fill="white", colour=None),
        plot_background=element_rect(fill="white", colour=None),
        panel_border=element_rect(colour="#444444", fill=None, size=0.8),
        panel_grid_major=element_line(colour="#DDDDDD", size=0.4),
        panel_grid_minor=element_blank(),
        axis_ticks=element_line(colour="#444444", size=0.5),
        axis_text=element_text(size=base_size - 1, family=base_family, colour="#222222"),
        axis_title=element_text(size=base_size, family=base_family, colour="#222222", face="bold"),
        strip_background=element_rect(fill="#F0F0F0", colour="#444444", size=0.6),
        strip_text=element_text(size=base_size - 1, family=base_family, colour="#222222"),
        legend_background=element_blank(),
        legend_key=element_blank(),
        legend_text=element_text(size=base_size - 1, family=base_family),
        legend_title=element_text(size=base_size - 1, family=base_family),
        plot_title=element_text(size=base_size + 1, family=base_family, colour="#111111"),
    )


def scale_fill_publication(**kwargs):
    """Discrete fill scale using PUBLICATION_COLORS."""
    return scale_fill_manual(values=PUBLICATION_COLORS, **kwargs)


def scale_color_publication(**kwargs):
    """Discrete colour scale using PUBLICATION_COLORS."""
    return scale_color_manual(values=PUBLICATION_COLORS, **kwargs)


# British-English alias
scale_colour_publication = scale_color_publication
