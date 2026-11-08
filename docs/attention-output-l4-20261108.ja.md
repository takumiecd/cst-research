# Colab L4での全Attention・FFN成立確認

[English](attention-output-l4-20261108.md) | 日本語

九工大4090が接続不能となり、ユーザーはColab L4での実行を許可した。
元serverの状態不明jobは再起動・重複投入しない。4090結果の継続・付け替えではなく、
別登録のfresh L4 cohortとする。host共通poolをworker1台で使い、正しいaccount・
実NVIDIA L4、隔離source、hash照合回収、VM停止を確認する。

torchcst数値sourceは符号付きFP32周期中心修正を含む
`df24415afe442b9916e902d0f590fa5207016f66`を維持。consumerはL4の明示識別と
本protocol記録のみ追加し、提出前に全commit/byteを固定する。Colabにある実際の
Torch/CUDA/Pythonを記録し、system packageを継承するjob内venvを使う。
学習receiptとL4検証のruntime同一性を要求し、GPU/runtime間の中央値を混ぜない。

モデル・データ・recipe・seed・予算は固定した幅512計画と同じ。pre-LN4block、8head、
context128、FFN倍率4、batch2、accumulation4、seed17、c8、minimum-001/q0、
FP32/TF32無効、基準LR3e-4・全Atom群×16、Adam(.9,.95)、decay0、clip1。
A=ceil(8√(d_in*d_out))で、正方4096Atoms・FFN8192Atoms。
既存TinyStories/GPT2の凍結token archiveを使い、manifest SHA256は
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。
train/DEVのみで、confirmation採点・生成・HPO・下限探索はしない。
要求1Mtoken＝976更新/実999424token、data/LR horizon20Mtoken。
fresh Dense、FFN、QKV、QKV+FFN、QKVO、QKVO+FFN（CST0/8/12/20/16/24箇所）を比較。
token embedding/共有語彙と学習位置embeddingはDense、mask/softmax/SDPAとLayerNormも
元の演算を保つ。

学習提出前に、L4検証jobで既存scope/CUDA回帰223件と、levels5構成・stream64の
24site検証を実施する。実shapeの全CPU FP64 Y/dX/全Atom VJP、Gauss積分、Oを含む
因果性・手計算Attention、全site実更新と保存後の次更新、同一勾配optimizer/moment/clockを
確認する。全model/backend/shape/kernel既定値/source/runtimeの照合と既存数値基準を
維持し、失敗/OOM/非有限/timeoutでその試行を保全して停止。自動再試行・基準緩和・
別hardwareへのfallbackはしない。

検証job成功とhash照合後、条件ごと6個のwhole-arm jobを同じworker1のqueueへ提出。
各jobはfresh source・build・初期化、対応する厳密な検証artifact、976更新と独立した
学習後checkpoint監査を含む。DEV厳密再現、全CST mapの少数行FP64 VJP、2回の独立復元と
次更新/state一致を確認。各jobの実GPU UUID/runtimeを記録し、同じVMを再利用しても
別job・host負荷のため時間比較は探索的とする。全checkpoint payloadを検証済みpool結果
archiveへ保存。転送前にhardlinkをbyte同一の通常fileへ実体化し、前後hashを記録する。
最終・milestone checkpointを捨てない。必要なraw datasetとconsumer sourceは、ignoredな
外部driverへhash付き入力archiveとして埋め込み転送し、torchcstへcommitしない。

train/DEV CE・PPLを実token数・wall timeで重ね、warmup後の全時間sampleと
allocated/reservedピークを保全。既存learnerのzero_grad/LR設定・初期同期後から、
転送、4回のforward/backward、clip、optimizer、終了同期までの時計を維持する。
setup、評価、probe、checkpoint IOは計時外でwall timeには含む。
allocatorピークは全GPU processメモリではなく、pool時間もthroughputではない。
1seed/約1Mtokenのdevelopment挙動であり、収束・精度同等・非劣性・最小Atom数・
GPU間byte一致・普遍的速度の結論にはしない。

## 実行準備

固定したconsumerは `10840baa8ce7f3570e7bf895f0618707b9f36652`。
学習開始前に2回の準備失敗を保全した。最初はvenvのensurepipが失敗し、
次はCUDA回帰174件通過後、構文検査がC++17を固定し、C++20を要求する
Torch2.14のヘッダで停止した。前者は専用venvの
`--without-pip --system-site-packages`、後者は検査だけのコンパイラ指定を
変更したoperator `b393ac653f3b3893c378acda1c1752919c1539d1` で修正する。
数値計算のソースは `df24415a` と同一。両失敗を残し、新しい実行では
223件と6つのモデル資格検査を許容誤差を緩めず全てやり直す。
観測したL4環境はTorch2.14.1+cu130、CUDA13.0、Python3.13.16。
資格検査も学習もまだ完了していない。

## 先に行う短期速度比較

長い学習より前に速度を知りたいという依頼に対応する。上記の全資格検査を
通過した同じGPU割当・環境で、既存の `ffn_run` を使い、1つのpool job内で
Denseと全24か所QKVO+FFN CSTを測る。各40更新（40,960 tokens）、最初の10更新を
warmupとして除いた30更新の時間を保持し、最終checkpointを保存する。
初期化・seed・batch・勾配累積・20M-tokenのデータ順/LR horizon・optimizer・
数値設定は同じ。学習前に新しくコンパイルしたbinaryの独立VJPを検査し、
学習後にcheckpointを監査する。監査の期待する更新数・token数・geometry行数
だけを40/40960/8に変える。この短期実行には1M-token milestoneを要求しない。

更新時間の中央値ms、その時間で1更新1024 tokensを処理するthroughput、
ピークallocated/reservedを報告する。ソース準備・コンパイル・評価・監査は
更新タイマーの外。Dense→CSTの1順序・短い初期軌道なので、学習済みモデルや
全幅の速度、精度・収束の比較には一般化しない。6条件1M-tokenの学習はまだ
投入していない。

その後、先に知りたい速度はN8192/16384・R256・A8Nの正方形Linear単体だと
依頼者が明確にした。上記40更新のLM比較は準備だけで投入していない。
既存Linear workerでDense・levels・levels_stream64を、10 warmup＋51計測更新、
逆順2回で比較する。N2048と初期・更新後VJPの検査を通す。
この演算子benchmarkをLM研究と分け、LM精度の結果とはしない。
現在のL4全資格検査は継続し、長い学習jobはまだ投入していない。
