# TinyStories FFNでの半径包絡tau探索

[English](ffn-radius-fraction-tau-20261109.md) | 日本語

新しい学習結果を見る前に固定した探索プロトコル。
公開torchcstのradius-envelopeで振幅・radius_fraction・中心を別々に学習し、
振幅に依存する下限への勾配も通す。U=50、L=.5+49.5 exp(-(a/tau)^2)、
r=L^(1-f) U^f、f=radius_fraction。定数を準備する明示Planは写像と全勾配を維持する。
その小さな速度改善パイロットと精度結果・公開デフォルト採用は別。
operator443ee84cを凍結し、runtimeはtorchcstに置く。

前回と同じTinyStories/GPT2トークンを使い、manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`、
実バイトとstory単位の分離宣言を検証する。選択はDEV8192予測対象のみ。
confirmationは選択に使わない。幅512、4pre-normブロック、8head、context128、
expansion4、FFN8射影をCST化。各射影A=8 sqrt(512*2048)=8192。
Attention・embedding・共有語彙headはDense。Parameter/物理行FP32、写像中間FP64、TF32無効。
通常重みのnamed stream、中心stream、振幅Gaussian乱数seedを共通化し、
各tauで公開のmoment初期化を使う。目標stdはup.02、down.02/sqrt(8)。
振幅・半径の実分布は変わり得るため、同一の物理重みとは主張しない。
各層の初期化目標・実現誤差を記録。fraction0、振幅上限1、半径[.5,50]。
実行中のsourceを変更しない。

短期はtau .003,.01,.03,.1,.3、seed17、各256更新/262144tokensと、同じ停止点の
新規Dense。最終DEV CEが有限の下位2条件を選択（同率は小さいtau）。
失敗・非有限条件は保存し選外。選択2条件を最初から976更新/999424tokensで
seed17,29,43実行し、新規Denseも同じ長予算で3seed。
長めの追試であり、収束・非劣性の確証ではない。DEV選択と3seedから普遍的最適tauは言えない。
正しさの問題なく2条件以上が完了した場合にのみ次段階へ進み、それ以外は原因を分析する。

batch2×128×accumulation4=1024tokens/更新。全条件20M-tokenのデータ順序と
スケジュールhorizonを維持し、warmup200更新。AdamW baseLR3e-4、betas(.9,.95)、
eps1e-8、decay0、globalclip1。振幅・中心はbaseLR16倍、radius_fractionは1倍を
公開CSTAdamWのfield_lrsで与え、同じ比率で減衰。LR・Atom数・深さ・fraction初期値を
同時探索しない。DenseはbaseLRの通常AdamW。

学習前に実CUDAの半径写像/物理行回帰、各tauの全モデル写像値・VJP一致、
同時に保持した2forwardの独立backward、全箇所の有限な更新とfractionの変化を検証。
tau.03ではmodel/AdamWを保存復元し、次のParameter・moments・step完全一致を検証。
新consumerのsourceに紐付いた新receiptを使い、旧receiptで新構成を承認しない。

train/DEV CEを更新数・token数・経過時間で重ねて描く。同期した完全更新時間と
allocated/reserved peakはaccumulation・転送・clip・optimizerを含み、評価・probeを除く。
準備・全wall時間を別記録。step0、1..5、20刻み、最終で各層の半径、fractionと端点比率、
振幅binごとの半径/fraction、固定probeのlatent各勾配を記録。
第1列勾配はdLoss/dfであり、物理偏微分dLoss/drではない。
Graph・GPUプロセス全体ピークは主張しない。平均だけでなく振幅別の推移も描く。

raw source/log/checkpoint/tokenはignored/externalへ。job/source/data/archive hashを残し、
取得したcompactファイルを検証してから完了とする。旧Polar曲線は当時のsourceの文脈に
留め、新しい同条件の速度比較に流用しない。
