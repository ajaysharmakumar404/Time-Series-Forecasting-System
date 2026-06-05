import matplotlib.pyplot as plt

def plot_results(actual, predicted):

    plt.figure(figsize=(15,6))

    plt.plot(
        actual.index,
        actual.values,
        label="Actual"
    )

    plt.plot(
        predicted.index,
        predicted.values,
        label="Forecast"
    )

    plt.title(
        "Stock Price Forecast"
    )

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "outputs/forecast.png",
        dpi=300
    )

    plt.show()