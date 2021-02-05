# %%
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display
from tqdm.auto import tqdm
from scipy import signal


# %%
def smooth(array, window=15, poly=3):
    return signal.savgol_filter(array, window, poly)


def plot_response(x, *ys, ylabel=None, legend_labels=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if legend_labels is None:
        legend_labels = [None] * len(ys)
    for y, lbl in zip(ys, legend_labels):
        ax.plot(x, smooth(y), label=lbl)

    ax.set_xlim((x.min(), x.max()))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=x.max()))
    ax.set_xlabel(r'% of Deep Knee Bend')
    ax.set_ylabel(ylabel)


def plot_bounds(x, y_train, ax=None):
    if ax is None:
        ax = plt.gca()
    avg = smooth(y_train.mean(axis=0))
    sd2 = 2 * y_train.std(axis=0)
    # err = np.zeros(len(y_pred))
    ax.fill_between(
        x,
        smooth(avg - sd2),
        smooth(avg + sd2),
        color='r',
        alpha=0.3,
        label=r'$\pm 2\sigma$'
    )


def create_plots(n_rows,
                 n_cols,
                 regressor,
                 resp_name,
                 fig_dir,
                 force_clobber=False):
    fig_dir.mkdir(parents=True, exist_ok=True)
    resp_str = resp_name.replace('_', '-')
    save_dir = fig_dir / resp_str
    save_dir.mkdir(exist_ok=True)

    tim = regressor.train_results.collect_response('time')[0]
    y_train = regressor.train_results.collect_response(resp_name)
    y_test = regressor.test_results.collect_response(resp_name)
    y_pred = regressor.fit(resp_name).predict()

    n_plots = int(n_rows * n_cols)
    n_img = np.ceil(len(y_pred) / n_plots)
    existing_plots = list(save_dir.glob('*.png'))
    if force_clobber or (n_img < len(existing_plots)):
        [p.unlink() for p in existing_plots]
    splits = np.arange(0, len(y_test), n_plots)[1:]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 30))
    combo = zip(np.array_split(y_test, splits), np.array_split(y_pred, splits))
    pbar = tqdm(total=len(splits), desc=f'Plotting {resp_name}')
    lbls = ['Simulated', 'Predicted']
    ylbl = resp_name.replace('_', ' ').title()
    for idx, (y_test_sp, y_pred_sp) in enumerate(combo):
        ax_pbar = tqdm(axs.ravel())
        for ax, y_t, y_p in zip(ax_pbar, y_test_sp, y_pred_sp):
            plot_bounds(tim, y_train, ax=ax)
            plot_response(tim, y_t, y_p, legend_labels=lbls, ylabel=ylbl, ax=ax)
        ax_pbar.close()

        ax.legend(loc='best')
        save_path = save_dir / '-'.join((resp_str, str(idx)))
        fig.savefig(save_path, bbox_inches='tight')
        pbar.update(1)
        [ax.clear() for ax in axs.ravel()]

    pbar.close()
    plt.close(fig)


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
    ax2.grid(False)
    fig.canvas.draw()
    yticks = ax2.get_yticks()
    yticks = np.round(yticks / ysum * 100).astype(np.int)
    labels = [str(yt) + '%' for yt in yticks]
    ax2.set_yticks(ax2.get_yticks().tolist())  # Workaround for avoiding warn
    ax2.set_yticklabels(labels)
    return (bar, lin), (ax, ax2)


class ImageViewer:

    def __init__(self, top_dir) -> None:
        top_dir = Path(top_dir)

        self.group_dir_list = [x for x in top_dir.iterdir()
                               if x.is_dir()]
        self.current_group_dir = self.group_dir_list[0]
        self.group_dir_names = [x.name for x in self.group_dir_list]
        self.resp_idx = 0

        self.layout = Layout(
            display='flex',
            flex_flow='row',
            align_items='stretch',
            justify_content='space-between',
            width='100%'
        )

        self.viewer = widgets.Image(
            format='png',
        )

        self.prv_button = widgets.Button(
            description='',
            disabled=False,
            button_style='',
            tooltip='Previous image',
            icon='arrow-left'
        )

        self.group_dropdown = widgets.Dropdown(
            options=list(zip(self.group_dir_names, self.group_dir_list)),
            value=self.group_dir_list[0],
            description='Group:',
            disabled=False
        )

        self.learner_dropdown = widgets.Dropdown(
            description='Regressor:',
            disabled=False
        )

        self.response_dropdown = widgets.Dropdown(
            description='Response:',
            disabled=False,
        )

        self.scrubber = widgets.IntSlider(
            value=0,
            min=0,
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

        self.selection_ctrl_box = widgets.Box(
            children=(
                self.group_dropdown,
                self.learner_dropdown,
                self.response_dropdown),
            layout=self.layout
        )

        self.scrub_ctrl_box = widgets.Box(
            children=(self.prv_button, self.scrubber, self.nxt_button),
            layout=self.layout
        )

        self.prv_button.on_click(self.on_prv_button_clicked)
        self.nxt_button.on_click(self.on_nxt_button_clicked)
        self.scrubber.observe(self.on_scrubber_change, names='value')

        self.group_dropdown.observe(
            self.on_group_dropdown_change,
            names='value'
        )
        self.learner_dropdown.observe(
            self.on_regr_dropdown_change,
            names='value'
        )
        self.response_dropdown.observe(
            self.on_resp_dropdown_change,
            names='value'
        )
        children = [self.viewer, self.selection_ctrl_box, self.scrub_ctrl_box]
        self.gui = widgets.VBox(children)
        self.update_regressor_dir_list()
        self.scrubber.max = len(self.im_list) - 1

    # List cascade
    def render_img(self):
        im_path = self.im_list[self.scrubber.value]
        with open(str(im_path), 'rb') as file:
            self.image = file.read()
        self.viewer.value = self.image

    def update_img_list(self):
        self.im_list = list(self.current_resp_dir.glob('*.png'))
        self.render_img()

    def update_resp_dir_list(self):
        self.resp_dir_list = [x for x in self.current_learner_dir.iterdir()
                              if x.is_dir()]
        self.resp_dir_names = [x.name for x in self.resp_dir_list]
        self.current_resp_dir = self.resp_dir_list[self.resp_idx]
        new_opts = list(zip(self.resp_dir_names, self.resp_dir_list))
        self.response_dropdown.unobserve(self.on_resp_dropdown_change, 'value')
        self.response_dropdown.options = new_opts
        self.response_dropdown.observe(self.on_resp_dropdown_change, 'value')
        self.response_dropdown.value = self.current_resp_dir
        self.update_img_list()

    def update_regressor_dir_list(self):
        self.learner_dir_list = [x for x in self.current_group_dir.iterdir()
                                 if x.is_dir()]
        self.learner_dir_names = [x.name for x in self.learner_dir_list]
        self.current_learner_dir = self.learner_dir_list[0]
        self.learner_dropdown.options = list(
            zip(self.learner_dir_names, self.learner_dir_list)
        )
        self.learner_dropdown.value = self.current_learner_dir
        self.update_resp_dir_list()

    # Dropdown cascade
    def on_resp_dropdown_change(self, change):
        self.current_resp_dir = change['new']
        self.resp_idx = self.resp_dir_names.index(self.current_resp_dir.name)
        self.update_img_list()

    def on_regr_dropdown_change(self, change):
        self.current_learner_dir = change['new']
        self.update_resp_dir_list()

    def on_group_dropdown_change(self, change):
        self.resp_idx = 0
        self.current_group_dir = change['new']
        self.update_regressor_dir_list()

    # Scrubber tools
    def on_nxt_button_clicked(self, b):
        self.scrubber.value += 1

    def on_prv_button_clicked(self, b):
        self.scrubber.value -= 1

    def on_scrubber_change(self, change):
        self.render_img()

    def show(self):
        display(self.gui)

    def __call__(self):
        return self
