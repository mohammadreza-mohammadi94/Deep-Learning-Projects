import logging
import matplotlib.pyplot as plt

def setup_logging(log_file="app.log"):
    """
    Setup logging configuration
    Configure logging to output messages to both a log file and the console.

    Parameters:
    ----------
    log_file : str, optional
        Name of the log file to which logs will be written (default is "app.log").
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Set logging for matplotlib
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    return logging.getLogger(__name__)


def save_plot(data, labels, title, xlabel, 
              ylabel, filename, linestyle=None):
    """
    Generate and save a plot with the provided data and formatting options.

    Parameters:
    ----------
    data : list of array-like
        List containing sequences of values to be plotted.

    labels : list of str
        Corresponding labels for each sequence in `data`.

    title : str
        Title of the plot.

    xlabel : str
        Label for the x-axis.

    ylabel : str
        Label for the y-axis.

    filename : str
        Path where the resulting plot image will be saved.

    linestyle : list of str, optional
        List of matplotlib-compatible line styles for each data series.
        If not provided, defaults to solid lines ('-').
    """
    plt.figure(figsize=(15, 6))
    for d, label, ls in zip(data, labels, linestyle if linestyle else ['-'] * len(data)):
        plt.plot(d, label=label, linestyle=ls)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
