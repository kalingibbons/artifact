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
        sm = ScalarMappable(cmap=colormap, norm=plt.Normalize(0, max(data_color)))
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

