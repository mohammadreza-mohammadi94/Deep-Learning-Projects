import logging
import matplotlib.pyplot as plt

def setup_logging():
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_plot(data, labels, title, xlabel, ylabel, filename, linestyles=None):
    """
    Generate and save a plot with the provided data and formatting options.
    """
    plt.figure(figsize=(10, 6))
    for d, label, ls in zip(data, labels, linestyles if linestyles else ['-'] * len(data)):
        plt.plot(d, label=label, linestyle=ls)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()