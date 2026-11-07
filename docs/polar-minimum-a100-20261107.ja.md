# 最小半径へ落ちるPolar下限のスクリーニング

[English (canonical)](polar-minimum-a100-20261107.md) | 日本語

ユーザーはPolar派生研究のタスクに同じTinyStories小型LMのFFNを選びました。
下限形状を先に、移動量依存の更新則を次に調べます。ここは第一段階で、境界だけを
変更し、第二段階の更新則はまだ実装しません。

既存の公開宣言で提案式を表せます。`radius_envelope="rational_zero_opening"`、
`birth=minimum=.5`、`zero_opening_scale=epsilon` とすると、
`L=.5+(U0-.5)*epsilon²/(a²+epsilon²)`、`U=U0`です。ゼロで厳密に合流し、
離れるとLが素早く.5へ近づきます。新しいパラメータ条件であり、native kernelの
新設やruntime実装のコピーではありません。

確保済みA100 MIG3g.40GBで、新しいrational/birth4対照とbirth.5のepsilon=.001/.003を
すべて初期q0で比較します。各400更新/409600トークン、LR/データ順序は20Mトークン
予定を保ちます。幅256、pre-LN4ブロック、FFN4N、c8/各4096Atom、8射影、seed17、
context128、batch2/累積4、FP32/TF32無効、AdamW(.9,.95)、decay0、clip1、
baseLR3e-4・Atom全体LR16倍を共通にし、Attention・埋め込み・出力はDenseです。
同じtoken manifest SHA256は
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。
DEVのみを用い、confirmation/生成による選択や自動長期延長は行いません。

A100の現在のTorch2.6.0+cu126は4090の2.11.0とは別runtimeです。各条件を学習前に
A100で独立求積、実FFN両方向の全Atom VJP、optimizer moment/再開、実タスク/再開で
検証します。全8射影を同じDense二次モーメントへ誤差.5%以内で再校正し、振幅と
clippingレポートを保存します。到達不能なら棄却し、代替条件で黙って学習しません。

学習/DEV、毎更新のtorus最短中心移動距離を前時点の物理半径で割った値、振幅bin別の
半径/q/個数、勾配、更新全体の時間、準備、allocated/reservedピークを保存します。
CPU診断は同期時間の後に行い、計測付き総経過時間には含めます。中心移動量は観測のみで、
更新やRNG状態を変更しません。学習後checkpointと再開更新を監査し、compact結果の
取得とハッシュを検証、checkpoint/dataはリモートに保持します。1seed探索で最適epsilon、
収束、最小容量、デバイス間の速度優位は主張しません。第二段階は別の数式・固定対照・
検証記録を用意します。
