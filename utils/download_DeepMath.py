from datasets import load_dataset
import pandas as pd


def main(output_path: str = "DeepMath-103k.csv", split: str = "train") -> None:
    """Save a specific split of DeepMath-103K to CSV."""
    # Specify split to get Dataset instead of DatasetDict
    ds = load_dataset("zwhe99/DeepMath-103K", split=split)
    # pandas DataFrame으로 변환 후 CSV 저장
    df = ds.to_pandas()
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()