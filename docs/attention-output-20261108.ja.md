# 幅512における全Attention射影・FFNの成立確認

[English](attention-output-20261108.md) | 日本語

torchcst PR154・cst-research PR34のmerge後、GPU実行前に準備。
torchcst `b14db87d34c1d6ba788517780528932473970505`（明示LM levels/stream64対応、
既存CUDA算術は不変）を使う。consumer sourceは実行前のdeploy manifestで固定する。
許可された4090を1台使い、実機/runtime、隔離source、SM8.9 build、標準nohup、
所有process receipt、hash照合した回収を確認。A100/Colabの追加確保はしない。

## 固定する実験族

幅512、pre-LayerNorm4block、8head（head次元64）、FFN倍率4、context128、batch2、
accumulation4、seed17。freshなDense、FFN単独、QKV単独、QKV+FFN、QKVO、
QKVO+FFNを比較し、CST site数は順に0/8/12/20/16/24。全blockが対象。
Q/K/Vは独立mapであり3Nの単一mapへ変更しない。
Atomは各mapで`ceil(8 sqrt(d_in d_out))`、正方4096・FFN up/down8192。
線形予算の成立を見る研究で、最小Atom数や漸近下限を求めない。幅1024/2048は後段階。

QKVOは残差へ戻すAttention出力射影Oも含む。初期化目標分散はOが`.02^2/(2*4)`、
Q/K/VとFFN-upが`.02^2`、FFN-downはOと同じ残差分散。公開variance-scaling初期化を
q0、rational_zero_opening/birth.5/tau.001、amplitude_max1/w_c.3で使用。
置換しない通常Parameterのnamed streamを全条件で一致させる。token embeddingと
語彙出力はDense共有、位置embeddingもDense。LayerNorm、活性化、causal SDPA、
softmax、biasの演算を保つ。幅計画の残りの全行列段階は実装・検証済みとはしない。

FP32 IEEE/TF32無効、AdamW基準LR3e-4、betas(.9,.95)、decay0、clip1、全Atom群LR×16。
data/LR horizonは20Mtokenで固定。要求1Mtokenに対し各条件976更新・実999424token。
以前と同じTinyStories/GPT2 streamのmanifest SHA256は
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。
train/DEVのみ使い、confirmation・生成・LR/半径探索は行わない。
失敗でその凍結試行を止め、自動再試行・数値基準の緩和はしない。

## 学習前の独立検証

6shape・両保持方式のscope/全勾配GPU testを実行する。既存`attention_qualification`を
拡張し、別のqualification/学習loopを作らない。学習するmodel kindごとに、
実shape・256token行で全CPU FP64 Fourier Y/dX/全Atom VJP、独立profile積分、
同一勾配optimizer/保存、手計算causal AttentionのY/dX/全QKVO勾配と未来勾配の厳密0、
全モデルの実更新と保存・再開後の次更新を確認する。全CST siteが有限非0勾配を受け
変化する必要がある。24siteでlevels_stream64も別途検証するが、学習にはlevelsを使用。
既存のmodel source/shape/kernel/recipeに加え、全config/head/backendの同一性を要求する。

学習後は各checkpointを独立復元してDEVを厳密再現し、学習済みの全CST層を少数token行の
全CPU FP64 VJPと比較。2回の独立復元から次のタスク更新を一致確認する。
checkpointはGPU上に保全しhashを記録、compact logを回収する。

## 測定と結論の範囲

train/DEV CE・PPLを実token数・wall timeで重ねる。warmup10更新後の全時間sampleと
allocated/reservedピークを記録。既存learnerはzero_grad/LR設定・初期同期後から計時し、
転送、4回のforward/backward、clip、optimizer、終了同期を含む。この時計を維持し、
単体Linearの時計と同一視しない。評価・geometry/勾配probe・checkpoint IOはupdate計時外、
wall timeには含む。setup/buildは別記。allocatorピークは全GPU processメモリではない。
Q/K/V/O/FFNの各roleについて診断を残す。

1seedのdevelopment成立確認であり、損失低下は学習の成立を示すが、収束・精度同等・
非劣性を示さない。fresh Denseを対照とし、全失敗・全条件を公開する。
長めの3seed確認、大規模幅、語彙/embedding置換は後続の固定した段階として扱う。
