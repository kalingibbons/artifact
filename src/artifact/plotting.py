# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display


# %%
def pareto(heights, cmap=None, names=None):
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(which='major', axis='y')
    ax.set_axisbelow(True)
    ysum = np.sum(heights)

    # Create the bar plot
    bar_ticks = range(len(heights))
    data_color = [height / max(heights) for height in heights]
    colormap = plt.cm.get_cmap()
    colors = colormap(data_color)
    if cmap is not None:
        colormap = plt.cm.get_cmap(cmap)
        colors = colormap(data_color)
        sm = ScalarMappable(cmap=colormap,
                            norm=plt.Normalize(0, max(data_color)))
        sm.set_array([])
        bar = ax.bar(bar_ticks, heights, color=colors, edgecolor='k',
                     tick_label=names)
    else:
        bar = ax.bar(bar_ticks, heights, edgecolor='k', tick_label=names)

    # Format the bar axis
    ax.set_xticks(bar_ticks)
    if names is not None:
        plt.xticks(rotation=45, ha='right')
    ax.set_ylim((0, ysum))
    yticks = ax.get_yticks()
    if max(yticks) < 0.9 * ysum:
        yticks = np.unique(np.append(yticks, ysum))
    ax.set_yticks(yticks)
    ylim = ax.get_ylim()

    # Add an axes for the cumulative sum
    ax2 = plt.twinx()
    lin = ax2.plot(bar_ticks, np.cumsum(heights), '.-', color=colormap(0))
    ax2.set_ylim(ylim)
    ax2.spines['right'].set_color(colormap(0))
    ax2.tick_params(axis='y', colors=colormap(0))
    ax2.title.set_color(colormap(0))
    fig.canvas.draw()
    yticks = ax2.get_yticks()
    yticks = np.round(yticks / ysum * 100).astype(np.int)
    labels = [str(yt) + '%' for yt in yticks]
    ax2.set_yticklabels(labels)
    return (bar, lin), (ax, ax2)


class ImageViewer:

    def __init__(self, top_dir) -> None:
        top_dir = Path(top_dir)
        self.dir_list = list(top_dir.iterdir())
        self.dir_names = [x.name for x in self.dir_list]
        self.im_list = list(self.dir_list[0].iterdir())
        with open(self.im_list[0], 'rb') as file:
            self.image = file.read()

        self.layout = Layout(
            display='flex',
            flex_flow='row',
            align_items='stretch',
            justify_content='space-between',
            width='100%'
        )

        self.viewer = widgets.Image(
            value=self.image,
            format='png',
        )

        self.prv_button = widgets.Button(
            description='',
            disabled=False,
            button_style='',
            tooltip='Previous image',
            icon='arrow-left'
        )

        self.output_dropdown = widgets.Dropdown(
            options=list(zip(self.dir_names, self.dir_list)),
            value=self.dir_list[0],
            description='Output: ',
            disabled=False
        )

        self.scrubber = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.im_list) - 1,
            step=1,
            description='',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=False,
            readout_format='d',
            layout=Layout(flex='1 1 0%', width='auto')
        )

        self.nxt_button = widgets.Button(
            description='',
            disabled=False,
            button_style='',
            tooltip='Next image',
            icon='arrow-right',
        )

        self.control_box = widgets.Box(
            children=(self.prv_button, self.scrubber, self.nxt_button),
            layout=self.layout
        )

        self.prv_button.on_click(self.on_prv_button_clicked)
        self.nxt_button.on_click(self.on_nxt_button_clicked)
        self.scrubber.observe(self.on_scrubber_change, names='value')
        self.output_dropdown.observe(self.on_dropdown_change, names='value')
        children = [self.viewer, self.output_dropdown, self.control_box]
        self.gui = widgets.VBox(children)

    def render_img(self):
        im_path = self.im_list[self.scrubber.value]
        with open(str(im_path), 'rb') as file:
            image = file.read()
        self.viewer.value = image

    def on_nxt_button_clicked(self, b):
        self.scrubber.value += 1

    def on_prv_button_clicked(self, b):
        self.scrubber.value -= 1

    def on_scrubber_change(self, change):
        self.render_img()

    def on_dropdown_change(self, change):
        self.im_list = list(change['new'].iterdir())
        self.render_img()

    def show(self):
        display(self.gui)

    def __call__(self):
        return self
