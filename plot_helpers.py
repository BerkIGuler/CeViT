import matplotlib.pyplot as plt


def plot_test_stats(x_name, stats, methods, show=False):
    assert len(stats) == len(methods)
    fig = plt.figure()
    symbols = iter(["*", "x", "+", "D", "v", "^"])
    for stat in stats:
        try:
            symbol = next(symbols)
        except StopIteration:
            symbols = iter(["o", "*", "x", "+", "D", "v", "^"])
            symbol = next(symbols)

        kv_pairs = sorted(list(stat.items()), key=lambda x: x[0])
        x_vals = []
        y_vals = []
        for key, value in kv_pairs:
            x_vals.append(key)
            y_vals.append(value)

        plt.plot(x_vals, y_vals, f"{symbol}--")
        plt.xlabel(x_name)
        plt.ylabel("MSE Error (dB)")
        plt.grid()
    plt.legend(methods)
    if show:
        plt.show()
    return fig
