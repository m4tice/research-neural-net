import math
import numpy as np

from colorama import Fore, init
from matplotlib import pyplot as plt

from utils import sigmoid

init()

# THEORICAL BACKGROUND
# ∂x/∂y means the derivative of x with respect to y OR how x changes as y changes

# ---- Bias
# Bias lets the model produce the right answer even when the input alone can't get you there — it "shifts" the line/curve instead of just scaling it.
# Concrete failure case, no bias at all:
# Say x = 0. Then z = w*x = w*0 = 0, no matter what w is — w becomes completely powerless when x=0.
# No matter how you train w, if x=0, output is stuck at 0.5. If your target was y=1 or y=0, this node can never reach it — training is mathematically impossible without a b term to escape this trap.
# With bias, the fix is trivial: z = w*x + b. Now even at x=0, z = b, and you can set/learn b to push the output anywhere you want.
# Geometric picture (line-fitting intuition): z = wx is a line forced through the origin (0,0) — you can only rotate it, not move it. z = wx + b lets you also slide the whole line up/down. Real data almost never happens to pass through the origin, so without b, you're fitting with one hand tied behind your back.
# One-line summary: w controls how much the input matters; b controls the baseline output when the input contributes nothing (or is zero). Without b, every neuron is forced to output exactly sigmoid(0) = 0.5 whenever x=0 — completely inflexible, and unable to fit most real targets.

# ---- Single layer, single node (1x1). Pure Python, no libraries. ----
# a = sigmoid(w*x + b)

w = 0.3      # weight (a guess)
b = 0.0      # bias
x = 1.0      # input
y = 1.0      # target output
lr = 5.0     # learning rate (bigger because sigmoid gradients are naturally small)

training_statistic = []

for step in range(50):
    # ---------------- FORWARD ----------------
    z = w * x + b        # linear part

    # σ(z)
    a = sigmoid(z)       # squish through activation -> final output

    # ---------------- LOSS L (just for monitoring) ----------------
    loss = 0.5 * (a - y) ** 2

    # ---------------- BACKWARD ----------------
    # blame = (output - target) * how_sensitive_this_node_is
    # ∂L/∂z = ∂L/∂a * ∂a/∂z
    # ∂L/∂a = a - y
    # ∂a/∂z = a * (1 - a)
    blame = (a - y) * a * (1 - a)     # a*(1-a) is sigmoid's own derivative

    # gradient_for_weight = blame * this_node's_input
    # ∂L​/∂w = ∂L/∂z * ∂z/∂w = blame * x
    dw = blame * x

    # ∂L​/∂b = ∂L/∂z * ∂z/∂b = blame * 1
    db = blame * 1        # bias always "sees" input = 1

    # ---------------- UPDATE ----------------
    # update weights and bias using gradient descent - step in the direction of negative gradient (downhill)
    w = w - lr * dw
    b = b - lr * db

    print(f"step {step}: a={Fore.GREEN}{a:.4f}{Fore.RESET}  loss={Fore.RED}{loss:.4f}{Fore.RESET}  w={Fore.BLUE}{w:.4f}{Fore.RESET}  b={Fore.BLUE}{b:.4f}{Fore.RESET}")
    training_statistic.append((step, a, loss, w, b))


def conclude():
    print("\nWith w, b, x, y as follows:")
    print(f"    w = {Fore.MAGENTA}{w:.4f}{Fore.RESET}")
    print(f"    b = {Fore.MAGENTA}{b:.4f}{Fore.RESET}")
    print(f"    x = {Fore.MAGENTA}{x:.4f}{Fore.RESET}")
    print(f"    y = {Fore.MAGENTA}{y:.4f}{Fore.RESET}")
    print("...give the following output:")
    print(f"    z = {Fore.MAGENTA}{(w * x + b):.4f}{Fore.RESET}")
    print(f"    a = {Fore.MAGENTA}{sigmoid(w * x + b):.4f}{Fore.RESET}")


def plot_model(w, b, x, y):
    # x-axis: input values, y-axis: output values
    xs = np.linspace(0, x, 100)              # many points, not just 2

    # compute the linear combination z = w*x + b for each input value
    zs = [w*xi + b for xi in xs]

    # apply the sigmoid function to each linear combination to get the model output
    outputs = [sigmoid(zi) for zi in zs]

    # plot the model output and the target output
    plt.plot(xs, outputs, label='model output a (sigmoid value)')
    plt.plot([0, x], [y, y], label='target y')   # your orange line, kept as-is
    plt.xlabel('x')
    plt.ylabel('σ(z)')
    plt.title('Sigmoid of Linear Function')
    plt.legend()
    plt.grid(True)

def plot_training_progress(losses):
    steps = [stat[0] for stat in losses]
    a_values = [stat[1] for stat in losses]
    loss_values = [stat[2] for stat in losses]
    w_values = [stat[3] for stat in losses]
    b_values = [stat[4] for stat in losses]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(steps, a_values, label='σ(z)')
    plt.xlabel('Step')
    plt.ylabel('σ(z)')
    plt.title('Output σ(z) over Training Steps')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(steps, loss_values, label='loss', color='red')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss over Training Steps')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(steps, w_values, label='w', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Weight w')
    plt.title('Weight w over Training Steps')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(steps, b_values, label='b', color='green')
    plt.xlabel('Step')
    plt.ylabel('Bias b')
    plt.title('Bias b over Training Steps')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()


# Call the conclude function to print the final state
conclude()

# Call the plot_model function to visualize the model output
plot_model(w, b, x, y)

# Call the plot_training_progress function to visualize the training progress
plot_training_progress(training_statistic)

plt.show()