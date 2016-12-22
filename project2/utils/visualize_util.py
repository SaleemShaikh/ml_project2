"""
Visualization utils
"""

from .io_utils import get_plot_path
try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import colors as colors
    from matplotlib import cm as cmx
except ImportError:
    raise RuntimeError("Cannot import matplotlib")


def plot_multiple_loss_acc(tr_loss, te_loss, tr_acc, te_acc, epochs=None, show=False,
                           names=('train', 'test'), xlabel='epoch', ylabel=('loss', 'mis-class'),
                           linestyle=('dotted', '-'), color=('b', 'g'),
                           filename='default.png'
                           ):
    assert len(tr_loss) == len(te_loss)
    assert len(tr_acc) == len(te_acc)
    if epochs is None:
        epochs = range(len(tr_loss))

    fig, ax1 = plt.subplots()
    # Plot the loss accordingly
    ax1.plot(epochs, tr_loss, color=color[0], label=names[0], linestyle=linestyle[0])
    ax1.plot(epochs, te_loss, color=color[0], label=names[1], linestyle=linestyle[1])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel=ylabel[0], color=color[0])
    for tl in ax1.get_yticklabels():
        tl.set_color(color[0])

    ax2 = ax1.twinx()
    ax2.plot(epochs, tr_acc, color=color[1], label=names[0], linestyle=linestyle[0])
    ax2.plot(epochs, te_acc, color=color[1], label=names[1], linestyle=linestyle[1])
    ax2.set_ylabel(ylabel=ylabel[1], color=color[1])
    for tl in ax2.get_yticklabels():
        tl.set_color(color[1])

    leg2 = ax2.legend(loc=1, shadow=True)
    leg2.draw_frame(False)
    leg1 = ax1.legend(loc=1, shadow=True)
    leg1.draw_frame(False)

    plt.title(filename)

    if show:
        plt.show()
    plt_path = get_plot_path("train_test " + filename)
    plt.savefig(plt_path)
    plt.close()
    return plt_path

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def plot_multiple_train_test(train_errors, test_errors, modelnames, show=False, title='',
                             xlabel='', ylabel='', filename='', linestyle=('dotted', '-'),
                             xlim=[0, 200], ylim=[0.01, 0.5]):
    assert len(train_errors) == len(test_errors) == len(modelnames)

    x_factor = range(len(train_errors[0]))
    cmap = get_cmap(len(train_errors) + 1)

    numModels = len(train_errors)

    plt.figure(figsize=(5, 5))
    # plt.gcf().subplots_adjust(bottom=0.3)

    for i in range(numModels):
        col = cmap(i)
        plt.plot(range(len(train_errors[i])), train_errors[i], color=col, linestyle=linestyle[0], linewidth=2.0, label=modelnames[i])
        plt.plot(range(len(test_errors[i])), test_errors[i], color=col, linestyle=linestyle[1])

    axes = plt.gca()
    axes.set_ylim(ylim)
    axes.set_xlim(xlim)

    leg = plt.legend(loc=4, frameon=True)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='medium')

    if show:
        plt.show()

    plt_path = get_plot_path("train_test " + filename)
    plt.savefig(plt_path, dpi=400)
    plt.close()

    return plt_path


def plot_train_test(train_errors, test_errors, x_factor=None, show=False,
                    names=('train', 'test'), xlabel='', ylabel='', filename='',
                    color=('b', 'r'), linestyle='-',
                    plot_type=0):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot
    :param plot_type: 0 for smooth, 1 for scatter
    """

    if x_factor is None:
        x_factor = range(len(train_errors))

    if plot_type == 0:
        plt.plot(x_factor, train_errors, color=color[0], label=names[0], linestyle=linestyle)
        plt.plot(x_factor, test_errors, color=color[1], label=names[1], linestyle=linestyle)
    elif plot_type == 1:
        plt.semilogx(x_factor, train_errors, color='b', marker='*', linestyle=linestyle, label=names[0])
        plt.semilogx(x_factor, test_errors, color='r', marker='*', linestyle=linestyle, label=names[1])
    else:
        raise RuntimeError("Unidentified plot type, must be either smooth or scatter")

    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(filename)

    if show:
        plt.show()
    plt_path = get_plot_path("train_test_" + filename)
    plt.savefig(plt_path)
    plt.close()
    return plt_path
