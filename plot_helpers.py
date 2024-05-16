import matplotlib.pyplot as plt


def plot_test_stats(var_name, stats):
    x, y = [], []
    for key, value in stats.items():
        x.append(key)
        y.append(value)
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel(var_name)
    plt.ylabel("MSE Error (dB)")
    return fig
