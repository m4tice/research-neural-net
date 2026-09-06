from colorama import Fore, init
from utils import sigmoid
import matplotlib.pyplot as plt

init()

# ---- Multiple training examples instead of just one ----
# A step-like pattern: low x -> target 0, high x -> target 1
data = [
    (0.0, 0.0),
    (1.0, 0.0),
    (2.0, 1.0),
    (3.0, 1.0),
]

w, b = 0.1, 0.0
lr = 0.5
snapshot_every = 500

snapshots = []   # (epoch, w, b) captured every 500 steps, for the plot

for epoch in range(3000):
    dw_sum, db_sum, total_loss = 0.0, 0.0, 0.0

    # ---- snapshot BEFORE this epoch's update ----
    if epoch % snapshot_every == 0:
        snapshots.append((epoch, w, b))

    # ---- accumulate blame/gradient across ALL examples ----
    for x, y in data:
        z = w*x + b
        a = sigmoid(z)
        total_loss += 0.5*(a-y)**2

        blame = (a-y) * a*(1-a)
        dw_sum += blame * x
        db_sum += blame * 1

    n = len(data)
    dw = dw_sum / n     # AVERAGE gradient across the whole dataset
    db = db_sum / n

    w -= lr * dw
    b -= lr * db

    if epoch % 500 == 0:
        print(f"epoch {Fore.GREEN}{epoch:4}{Fore.RESET}: avg_loss={Fore.RED}{total_loss/n:.4f}{Fore.RESET}  w={Fore.BLUE}{w:.3f}{Fore.RESET}  b={Fore.BLUE}{b:.3f}{Fore.RESET}")

snapshots.append((3000, w, b))   # final trained state too

print(f"\nFinal: w={Fore.BLUE}{w:.4f}{Fore.RESET}  b={Fore.BLUE}{b:.4f}{Fore.RESET}\n")

print("--- Performance on TRAINING points ---")
for x, y in data:
    a = sigmoid(w*x+b)
    print(f"x={x}  target={y}  output={Fore.GREEN}{a:.4f}{Fore.RESET}")

print("\n--- Performance on UNSEEN points (never trained on these!) ---")
for x in [-1.0, 0.5, 1.5, 2.5, 4.0]:
    a = sigmoid(w*x+b)
    print(f"x={x}  output={Fore.GREEN}{a:.4f}{Fore.RESET}")

# ---------------- PLOT: evolution of the sigmoid fit every 500 epochs ----------------
xs_data = [x for x, y in data]
ys_data = [y for x, y in data]
xs_curve = [i/100 for i in range(-150, 450)]   # -1.5 to 4.5

plt.figure(figsize=(8, 6))
colors = plt.cm.viridis([i/(len(snapshots)-1) for i in range(len(snapshots))])

for (epoch, w_, b_), color in zip(snapshots, colors):
    ys_curve = [sigmoid(w_*x + b_) for x in xs_curve]
    plt.plot(xs_curve, ys_curve, color=color, label=f'epoch {epoch}')

plt.scatter(xs_data, ys_data, color='red', zorder=5, s=90, label='training data')
plt.axhline(0.5, color='gray', linestyle=':', linewidth=1)

plt.xlabel('x')
plt.ylabel('output')
plt.title('Evolution of sigmoid fit over training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()