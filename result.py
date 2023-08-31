import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

for fold in range(3):
    metrics = pd.read_csv(
        f"/kaggle/working/logs_f{fold}/lightning_logs/version_0/metrics.csv"
    )
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    g = sn.relplot(data=metrics, kind="line")
    plt.title(f"Fold {fold}")
    plt.gcf().set_size_inches(15, 5)
    plt.grid()
