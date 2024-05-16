import matplotlib.pyplot as plt


def plot_test_stats(var_name, stats, show=False):
    kv_pairs = sorted(list(stats.items()), key=lambda x: x[0])
    x_vals = []
    y_vals = []
    for key, value in kv_pairs:
        x_vals.append(key)
        y_vals.append(value)
    fig = plt.figure()
    plt.plot(x_vals, y_vals)
    plt.xlabel(var_name)
    plt.ylabel("MSE Error (dB)")
    plt.grid()
    if show:
        plt.show()
    return fig
