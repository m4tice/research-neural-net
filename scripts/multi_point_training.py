import math

from scripts.utils import sigmoid

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

for epoch in range(3000):
    dw_sum, db_sum, total_loss = 0.0, 0.0, 0.0

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
        print(f"epoch {epoch}: avg_loss={total_loss/n:.4f}  w={w:.3f}  b={b:.3f}")

print(f"\nFinal: w={w:.4f}  b={b:.4f}\n")

print("--- Performance on TRAINING points ---")
for x, y in data:
    a = sigmoid(w*x+b)
    print(f"x={x}  target={y}  output={a:.4f}")

print("\n--- Performance on UNSEEN points (never trained on these!) ---")
for x in [-1.0, 0.5, 1.5, 2.5, 4.0]:
    a = sigmoid(w*x+b)
    print(f"x={x}  output={a:.4f}")