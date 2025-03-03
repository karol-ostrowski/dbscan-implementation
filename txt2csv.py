import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], header=None, names=["a", "b", "c"])
df = df.drop(columns=["c"])
df.to_csv(sys.argv[2], header=False, index=False)