import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.size"] = 12


def plot_reaction_function():
    # Create separate x-arrays for negative and positive domains
    x_neg = np.linspace(-1, 0, 500, endpoint=False)  # negative part
    x_pos = np.linspace(0, 1, 500)  # positive part

    # Define the piecewise y-values
    y_neg = np.arctan(1.55 * x_neg)  # for x < 0
    y_pos = np.tan(0.785 * x_pos)  # for x >= 0

    # Create the plot
    plt.figure(figsize=(4.5, 2.5))

    # Plot each piece in its domain
    plt.plot(x_pos, y_pos,
             label='Underutilized', linewidth=3
             )
    plt.plot(x_neg, y_neg,
             label='Oversubscribed', linewidth=3
             )

    # Set axes limits
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    # Labels and legend
    plt.xlabel('Normalized Error (' + r'$e_{t\_prd}$' + ')')
    plt.ylabel('Reaction Function (' + r'$F$' + ')')
    #plt.title('Piecewise Function Plot')
    plt.grid(which='major', linestyle='-', linewidth=0.7, alpha=0.8)
    plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.legend()

    # Display the plot
    plt.savefig('results_cpu/reaction_function.svg')


# Example usage
if __name__ == "__main__":
    plot_reaction_function()
