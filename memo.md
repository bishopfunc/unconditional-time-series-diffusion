## トラブルシューティング
### 環境構築
- 個人的には`uv`が一番楽だと思う、他の実行環境でも動くはずなので適宜読み替えほしい
- M1 Macは`gluonts`内の`mxnet`のパッケージがダウンロードできないため、結構しんどそう
- 自分はそれがめんどうでLinux環境で実行している

使用環境
- 仮想環境 `uv`
- OS: `Ubuntu 24.04`

```bash
git clone ...
uv sync # これ一発でpyproject.tomlの内容を元に仮想環境が作成される
source .venv/bin/activate # 仮想環境を有効化
```
### error: Distribution mxnet==1.9.1 @ registry+https://pypi.org/simple can't be installed because it doesn't have a source distribution or wheel for the current platform
hint: You're on macOS (macosx_12_0_arm64), but mxnet (v1.9.1) only has wheels for the following platforms: manylinux2014_aarch64, manylinux2014_x86_64, macosx_10_13_x86_64
- 前提: `uv sync`実行時に発生
- 原因: `gluonts`内の`mxnet`がM1 Macに対応していない


### error: distribution `torch==1.13.1 @ registry+https://pypi.org/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform
- 前提: `python bin/train_model.py -c configs/train_tsdiff/train_uber_tlc.yaml --device cpu`実行時に発生
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
- 前提: `python bin/train_model.py -c configs/train_tsdiff/train_uber_tlc.yaml --device cpu`実行時に発生
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

### ValueError: invalid literal for int() with base 10: 'cpu'
- 前提: `python bin/train_model.py -c configs/train_tsdiff/train_uber_tlc.yaml --device cpu`実行時に発生
- 原因: `unconditional-time-series-diffusion/bin/train_model.py`の`devices`がGPU前提
- 対策: `unconditional-time-series-diffusion/bin/train_model.py`と`unconditional-time-series-diffusion/src/uncond_ts_diff/uncond_ts_diff.py`を以下のように修正
```python
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=[int(config["device"].split(":")[-1])] if "cuda" in config["device"] else None, # 修正
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=log_dir,
        gradient_clip_val=config.get("gradient_clip_val", None),
    )
```

### ModuleNotFoundError: No module named 'pykeops_cpp_a7c1d53916'
- 前提: `python bin/train_model.py -c configs/train_tsdiff/train_uber_tlc.yaml --device cpu`実行時に発生
- 原因: 同時に`g++ not found`のエラーが出る場合は、それが原因
- 対策: `sudo apt install -y build-essential`を実行して、`g++`をインストールする。`g++ vesion`を実行して、表示されればOK。
```bash

### KeyError: 'nvrtc'
- 原因: `CPU`で実行しようとしたが、`pykeops`がGPU前提であるため
- 対策: `unconditional-time-series-diffusion/src/uncond_ts_diff/arch/s4.py`のbackendを`CPU`に変更
```python
# 変更前
r = vandermonde_mult(v, x, l, backend="GPU")
# 変更後
device = "GPU" if v.is_cuda else "CPU"
r = 2 * cauchy_mult(v, z, w, backend=device)
```
もう1ヶ所も同様に変更


## 評価コマンド
```bash
python bin/guidance_experiment.py -c configs/guidance/guidance_uber_tlc.yaml --ckpt lightning_logs/version_0/best_checkpoint.ckpt
```

各種指標
```bash
2025-07-01 15:54:53,500 - /home/ubuntu/workspace/unconditional-time-series-diffusion/bin/guidance_experiment.py - INFO - Metrics for scenario 'none': {'MSE': 39.297214810155886, 'abs_error': 17125.078073501587, 'abs_target_sum': 82722.0, 'abs_target_mean': 13.155534351145038, 'seasonal_error': 3.986713170745807, 'MASE': 2.4960780407345173, 'MAPE': 0.5443996419231946, 'sMAPE': 0.9332610917379536, 'MSIS': 108.8980959138119, 'QuantileLoss[0.1]': 7593.1158324380485, 'Coverage[0.1]': 0.06504452926208652, 'QuantileLoss[0.2]': 11654.799939769127, 'Coverage[0.2]': 0.16841603053435114, 'QuantileLoss[0.3]': 14420.795004619647, 'Coverage[0.3]': 0.2981870229007634, 'QuantileLoss[0.4]': 16310.57931382019, 'Coverage[0.4]': 0.4193702290076336, 'QuantileLoss[0.5]': 17125.07800353485, 'Coverage[0.5]': 0.5063613231552163, 'QuantileLoss[0.6]': 17304.87530237807, 'Coverage[0.6]': 0.5874681933842238, 'QuantileLoss[0.7]': 16561.642788933226, 'Coverage[0.7]': 0.6765267175572519, 'QuantileLoss[0.8]': 14704.322952762406, 'Coverage[0.8]': 0.7806933842239187, 'QuantileLoss[0.9]': 10543.985448028408, 'Coverage[0.9]': 0.876590330788804, 'RMSE': 6.268749062624527, 'NRMSE': 0.4765104096344748, 'ND': 0.20701963290903977, 'wQuantileLoss[0.1]': 0.0917907670563822, 'wQuantileLoss[0.2]': 0.1408911769513446, 'wQuantileLoss[0.3]': 0.1743284132953706, 'wQuantileLoss[0.4]': 0.1971734159452164, 'wQuantileLoss[0.5]': 0.2070196320632341, 'wQuantileLoss[0.6]': 0.20919314453685925, 'wQuantileLoss[0.7]': 0.20020844260212792, 'wQuantileLoss[0.8]': 0.1777558926617152, 'wQuantileLoss[0.9]': 0.12746289316056683, 'mean_absolute_QuantileLoss': 14024.354954031553, 'mean_wQuantileLoss': 0.16953597536364634, 'MAE_Coverage': 0.38179247950240314, 'OWA': nan}
```