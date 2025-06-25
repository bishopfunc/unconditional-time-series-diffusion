### 環境
他の実行環境の人は適宜変更してほしい
- 仮想環境 `uv`
- OS: `Ubuntu 24.04`

```bash
git clone ...
uv sync # これ一発でpyproject.tomlの内容を元に仮想環境が作成される
source .venv/bin/activate # 仮想環境を有効化
```

### error: distribution `torch==1.13.1 @ registry+https://pypi.org/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform
- 原因: torchのバージョンが古い
- 対策:`pyproject.toml`の`requires-python = "==3.9"`に変更

### ModuleNotFoundError: No module named 'uncond_ts_diff'
- 原因: `src.uncond_ts_diff`がPYTHONPATHに含まれていない
- 対策: `PYTHONPATH`に`src`ディレクトリを追加する
```bash
cd src
export PYTHONPATH="$PWD:$PYTHONPATH"
cd ..
python bin/train_model.py -c configs/train_tsdiff/train_uber_tlc.yaml 
```

### NotImplementedError: Lags for H are not implemented yet.
- 原因: `unconditional-time-series-diffusion/src/uncond_ts_diff/utils.py`の`get_lags_for_freq`の小文字大文字判定のバグ
- 対策: `get_lags_for_freq`の`freq`の小文字大文字を無視するように修正
```python
def get_lags_for_freq(freq_str: str):
    offset = to_offset(freq_str)
    print(f"{offset=}, {offset.n=}, {offset.name=}")
    if offset.n > 1:
        raise NotImplementedError(
            "Lags for freq multiple > 1 are not implemented yet."
        )
    if offset.name.upper() == "H":
        lags_seq = [24 * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]
    elif offset.name.upper() == "D" or offset.name.upper() == "B":
        # TODO: Fix lags for B
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    else:
        raise NotImplementedError(
            f"Lags for {freq_str} are not implemented yet."
        )
    return lags_seq
```