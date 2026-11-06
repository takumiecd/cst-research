# 選択したFFN設定の再現

[English (canonical)](ffn-reproduction.md) | 日本語

選択したR8/R10/R11設定を新規に再現する手順です。所有者の実行中campaignを再開したり、
別GPUを自動的に検証済みと扱ったりはしません。
[研究計画](ffn-feasibility.ja.md)と[結果](ffn-feasibility-results.ja.md)に、探索の履歴と
未完了の確認状態を残しています。

## 両repoと実行環境を固定する

科学的な学習runは以下の不変ソースを使います。

- cst-research：`70e02d864e661d3a87cd8aea322b49ae3f4a1f8f`。
- torchcst：`2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440`。
- 実測環境：Python3.12.9、Torch2.11.0+cu126、CUDA12.6、NVIDIA GeForce RTX4090、
  24GB級、CC8.9。TF32無効、FP32学習、CPU4 threadです。

対応するTorch/CUDA build、CUDA compiler・headerを用意した独立環境で、operatorとconsumerを
明示的に固定してインストールします。

```bash
python -m pip install 'torchcst[cuda] @ git+https://github.com/takumiecd/torchcst.git@2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440'
python -m pip install 'cst-research @ git+https://github.com/takumiecd/cst-research.git@70e02d864e661d3a87cd8aea322b49ae3f4a1f8f'
python -m pip install 'tiktoken==0.12.0' numpy ninja
```

consumerの過去の`[torchcst]` extraは旧HIGGSソースを指すため、この研究には使いません。
compiler/header環境も保持してください。package versionだけではnative extensionのcompileを再現
できません。証拠locator内の固定bootstrap scriptに環境設定を残しています。別machineでは実際の
GPU・runtimeを記録し、学習前に検証します。実行中の学習ソースは編集しません。

## データを準備して検証する

```bash
python -m cst_research.experiments.ffn_data --output evidence/new-ffn-tokens
```

TinyStories revisionは`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`で、元のtrain/valid text、
GPT-2 encoding、packing前のstory単位重複除外、little-endian uint32を使います。
消費した正規化prefixをstreamする準備で、upstream全ファイルのdownload/digestは主張しません。
実行前にmanifestと実token/story-IDファイルのhashを両方確認します。

| Split | 保存token数 | Token SHA256 |
| --- | ---: | --- |
| Train | 20000001 | `9bfb0a333b8deb13d9f0aa9bcd64aeca4c7d512c9ac548acb8fdbe72bf3cc9ca` |
| Development | 8193 | `56656e2d90717be6919bc4d26f0b0095b3739c614ae4c071ab629142e4d0fd51` |
| Confirmation | 32769 | `2bcc516a29e7aa941edf633576720cf38da120266e094f6772b6f19e4fce17a8` |

記録したmanifest SHA256は
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`です。
独立データ監査で保存した全hashとstory集合の非重複を確認し、最初のupstream trainの重複しない
3 story・548 tokenを再現しました。このprefix replayはupstream corpus全体の検証ではありません。

## 形状と設定を厳密に合わせて検証する

以下は幅256・seed29の例です。新しい出力先を使います。

```bash
python -m cst_research.experiments.ffn_qualification \
  --cohort recipe --width 256 --expansion 4 --coefficient 8 \
  --q 0 --amplitude-max 1 --w-c 0.3 --radius-birth 4 \
  --feature-order contiguous --atom-lr-multiplier 16 \
  --output evidence/new-n256-seed29-qualification
```

`qualification.json`を確認し、両射影方向・実際の目標標準偏差・Y/dX/全Atom VJP・quadrature・
optimizer・model resumeが全て通ったことを確認します。SHA256を計算して保持し、次のコマンドに
明示的に渡します。別設定・形状・Atom数の検証では不十分です。N512 expansion4とN128 expansion16
には上記の明示的2048 operatorが必要で、対応するAtom数やeager FP32の範囲はこの設定で拡大しません。
記録したcampaignは、学習前にモデル全体でCSTの2 update preflightも要求します。

## 3条件を同じ学習量で実行する

`QUALIFICATION_SHA256`を検証したdigestに置き換え、`dense`・`none`・`cst`をそれぞれ
異なる新しい出力先で実行します。

```bash
python -m cst_research.experiments.ffn_run \
  --data evidence/new-ffn-tokens \
  --data-manifest-sha256 5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01 \
  --qualification evidence/new-n256-seed29-qualification/qualification.json \
  --qualification-sha256 QUALIFICATION_SHA256 \
  --output evidence/new-n256-seed29-cst --kind cst --seed 29 \
  --width 256 --expansion 4 --coefficient 8 \
  --q 0 --amplitude-max 1 --w-c 0.3 --radius-birth 4 \
  --feature-order contiguous --atom-lr-multiplier 16 \
  --batch 2 --accumulation 4 --tokens 5000000 \
  --schedule-tokens 20000000 --max-wall-seconds 7200
```

5M条件の前に、同じ引数で`--tokens 2048 --repeat-fixed-batches`と別のpreflight出力先を
指定し、CSTの2 updateを実行します。この固定batch preflightはconfirmation・生成を省略し、
heldoutの学習結果には含めません。

モデルは4 pre-LayerNorm block、4 head、context128、GELU、dropoutなしです。CST条件はFFN全8射影を
置換し、attention・embedding・tied outputはDenseです。`none`はFFNの残差への寄与を除外します。
初期Dense stdは0.02、残差出力は0.02/sqrt(8)で、公開variance-scaling初期化でCSTの重みenergy目標を
合わせます。CSTだけの実行時gainや正規化はありません。設定のscalar flagは3条件で全て同じにします。
Atom LR16はCST optimizer内でAtomの全4座標に適用し、通常Parameterの基本LR3e-4は維持します。
betas(.9,.95)、eps1e-8、decay0、global clipping1、warmup/cosineは固定runnerで定義します。

要求5M tokenは4882 update・実際の4999168 tokenに切り下げます。20M scheduleはデータのpermutation
horizonも決めるので、変更すると初期の学習軌道も変わります。この精度・resource比較では固定batchの
繰り返しやupdateごとの振幅サンプリングを有効にしません。部分状態や失敗を保持し、黙って再開・再試行・
学習量増加をしません。

固定した確認gridはseed29/43/71×幅128/256/512、expansion4、c8、3条件を各5Mで、新規27学習runです。
各caseのqualification・preflight・終了監査を通してから次へ逐次実行します。縦横比の確認である
N128 expansion1/16は別のseed17・1M開発pilotです。全seedを保持し、確認スコアから設定を調整しません。
これは学習量を限定した実験で、収束の約束ではありません。固定した有効性基準と主張の範囲は研究計画にあります。

## 監査・回収・報告

`$CST_RESEARCH_EVIDENCE_ROOT/ffn-feasibility-20261106`から、保存したsource manifest、
実際のlaunch/audit argv、raw summary/curve、checkpoint hash、監査scriptを辿ります。
`source-r8.tar` SHA256は
`d0ad97fbf6954f428b786f8cbd8ebbef23bb8310f18ccef1f4af7ffac139fdc3`で、
consumer全224ソースファイルをmanifestと照合しています。元のhost用wrapperは保持証拠であり、
汎用公開cluster launcherではありません。別のdeployment rootではwrapperを別版にし、
実行中の固定コピーは編集しません。

記録した学習済みcheckpoint auditor `audit-ffn-checkpoints-r8-v2.py` SHA256は
`25096a1faad73023a3ef70362e188d7a4f31cbd64359fcfea51d0676914040bc`です。
保存hash/counter、loadの契約、CEの独立再計算、学習済みCST全8射影のVJPを検査し、使い捨てコピーだけで
2 updateしてserialize後の次task stepとoptimizerを確認します。元の保存runを再開・上書きしません。
FFNなしには意図的に未使用のnorm2 Parameterがあり、修正したauditorはその厳密な集合だけ
勾配・stateの欠如を許します。大きなcheckpointはGPU hostに保持し、報告前に回収metadata archiveと
bundle内の全ファイルをmanifestと照合します。

別の[receiptを辿るreporting CLI](ffn-reporting.ja.md)を、consumer
`6eec101cf5092d319d280bd819ac4a97867e2004`またはreporterが変わっていない後続commitで使います。
独立したreporting環境に任意の`[reporting]` extraをインストールします。報告は学習ソースの移行や
再検証ではありません。requestには上記の実際のlearner/operator commit、全cohort、個別receipt/audit hashを
指定します。全曲線・seed、対応するCE差、両prompt sample、complete stepのallocated/reserved観測を保持し、
描画した全図を確認します。Parameter圧縮は、実測GPUメモリや速度の優位を意味しません。
