"""从 scikit-learn 的公开小数据集生成本实验的标准输入；无需私人数据。"""
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits


def main():
    digits = load_digits()
    out = Path(__file__).resolve().parents[1] / "data" / "digits_8x8.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, X=(digits.data / 16.0).astype(np.float32), y=digits.target.astype(np.int64))
    print(f"Prepared {len(digits.target)} samples: {out}")


if __name__ == "__main__":
    main()
