# CSTのFFN検証

[English (canonical)](ffn-feasibility.md) | 日本語

## 問いと現在地

`A=ceil(c sqrt(d_in d_out))`の係数を固定したまま、小型自己回帰言語モデルの
全ブロックでFFNの両射影をCSTへ置換して使えるかを調べます。最小Atom数やBig-Ωを
求める研究ではありません。準備・native検証・学習・独立した結果確認は別の段階です。
登録しただけで言語モデルの有効性を実証したとは扱いません。

初期構成はpre-LayerNormのTransformer 4ブロック、Attention 4ヘッド、幅256、
GELUのFFN `256→1024→256`、文脈128、dropoutなしです。CST条件では全8射影を
CSTにします。Embedding・Attention・重み共有した出力はDenseです。Dense FFN・
FFNなし条件にも、共有部分が完全一致する初期値系列、同じデータとトークン予算を使います。
FFNなしではFFNの残差寄与を省きます。Attentionだけでも学習できるため、loss低下だけでは
CST FFNが有用だったとは判断できません。

## Atom予算と初期化

まずc=8、q=0です。`N→4N→N`の各射影はA=16Nになります。Polar Atomは
4座標なので、FFNの両biasを含めると1ブロックで`128N+5N=133N`パラメータです。
Denseは`8N²+5N`です。幅N128/256/512、FFN中間幅512/1024/2048を確認します。
これはFFNのパラメータ数の線形性で、実行時間・GPUピーク・モデル全体がO(N)という主張では
ありません。Dense Attentionには二乗のパラメータ項が残ります。

optimizer作成とCUDA転送の前に公開`torchcst.init.variance_scaling_`を使います。
Denseの標準偏差0.02に対応させ、両条件で残差出力射影を`1/sqrt(2*layers)`倍します。
初期化reportと実測activation RMS・勾配を保存します。Polarのradius cell-massは
派生半径を通るtask勾配をdetachする既存契約を保ちます。CSTだけに実行時gainや正規化を
加えません。qは初期半径の割合であり、Atom密度でも学習中に固定する座標でもありません。

## タスク学習の前の検証

1. 独立した多項式の数値積分で小さな長方形を確認します。半径detachの既存契約に従って、
   振幅・中心のVJPも検証します。
2. CPU FP64のfactor・chunk参照でnativeのY・dX・全Atom全座標を、両射影方向・
   非連続入力・多トークンで比較します。数学的factorコードは共有しますが、CUDAの階層計算や
   native backwardは使いません。独立積分を置き換えるものではなく、組み合わせて確認します。
3. 制御した勾配によるCSTAdamW更新を公開optimizer wrapper参照と比較し、Parameter・
   moment・counterを調べます。保存・再読込後の次の制御した更新は厳密一致を求めます。
   再読込で進むAtomStateのcache世代は既存契約を別に検証し、全payload Tensor・row ID・
   他のmetadataは厳密一致させます。小さな自己回帰モデル全体でも保存状態と次のCE更新を
   FP32の許容差で確認します。
4. causal mask、条件間での共有初期値、CST FFNへの勾配、タスク更新を確認してから長期実行します。

最初は公開CSTLinear・CSTAdamWと内部Plan `GLOBAL_ACTIVE_GRAD4_NATIVE`、
明示した座標更新Plan `RADIUS_FLAT_TORUS_FUSED`を使います。既定の半径更新参照はCPU専用です。
これは実験の内部依存として記録し、公開dispatcherの既定採用とは扱いません。
現状の1024-site制約ではN512の2048幅射影が未対応なので、明示的な実装経路を拡張・
独立検証してから進みます。未対応shapeを表現能力の失敗とは扱いません。
縦横比1:1、1:4/4:1、1:16/16:1も限定確認します。縦横比を固定した幅の実験だけでは、
幾何平均と他の同次予算式を区別できません。

準備した拡張はtorchcst commit `2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440`
（[PR150](https://github.com/takumiecd/torchcst/pull/150)）の明示的な
`GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048`です。既存のrecipe項目・Planのguard・
native演算を保ちます。N512とN128のexpansion16を含む拡張12条件を検証してから、
そのsourceで学習します。初期R2の幅256 pilotは元のresearch/operator snapshotで続け、
途中で移行しません。

主実験のexpansion4の幅比較に加え、N128のexpansion1・16でもc=8を保ち、
development seed17・100万tokenに限ったpilotを行います。縦横比への適用を確認するもので、
最適な式や最小係数は求めません。その後、実行は所有者の指示でRTX4090だけに限定しました。
A100の小規模検証とCPUでのtoken準備は終了しており、追加実行は今回の計画に含めません。
保持するデバイス別の準備記録はRTX4090の結果と分けます。

## 現在の原因分析

R2の100万token・seed17 pilotは完了し、development CEはDense3.7273、CST4.1798、
FFNなし4.2307でした。CSTの初期FFN出力RMSはDenseに近い一方、縮小射影のRMSは
0.04–0.05付近に留まり、Denseは0.12–0.17へ増えました。容量・最適化・学習期間の
どれが差の原因かは確定していません。976更新のcosineも既に3e-5まで下がっており、
長いscheduleと比較して学習時間だけの効果を切り分けることはできません。

最初の追加計画では、N256のc8/16/32（各射影4096/8192/16384 Atoms）を、同じq0・
目標分散・基本LR・データ・学習token数で比較する予定でした。その後のownerとの議論で、
Atom数を増やす前に、後述するR7の半径・順序・Atom LRの限定比較を優先しました。
この比較で不十分なら係数の候補も残します。Atom予算ごとに同じDense二乗平均目標で
初期化し直し、実際の初期activationも記録します。実用設定を見つける限定比較で、
下限・Big-Ω研究の再開ではありません。

途中checkpointでも、LRとデータ順序の計画期間を共通の2,000万tokenにします。
`--schedule-tokens 20000000`でwarmup/cosineと全データの順序を固定し、`--tokens`で
限定実行の長さを指定します。これにより同じ設計の100万・500万・1,000万・2,000万tokenの
観測で、初期のLR・データ系列が揃います。旧R2の短いscheduleは別の観測として保持します。
現在のc8の長い曲線を記録してからAtom不足を判断し、Atom予算は同じcheckpointで比べます。
Atom数の変更では座標の幾何や各Atomの振幅も変わるため、改善だけで純粋な表現力不足を
証明したとは扱いません。

## 学習と証拠の計画

TinyStoriesのdataset revision・tokenizer・tokenファイルのハッシュを固定します。
物語単位で分割してからpackingし、trainとvalidationを分離します。次トークンへのshift・
評価window・promptを実行前に固定します。
初期データはrevision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`の元train/valid
テキストとtiktoken0.12.0のGPT-2 encodingです。train20,000,001、development8,193、
confirmation32,769 tokenを保持します。packing前に物語の完全一致によるsplit間重複を除きます。
実tokenファイル・encoding・消費した正規化済み物語prefixをハッシュ化します。prefixのハッシュを
元ファイル全体を取得したハッシュとは扱いません。prefixと物語IDは外部に保持し、
confirmationは各限定runの最後に評価します。

初期学習はFP32・TF32off、microbatch2×文脈128、累積4で1更新1,024 tokenです。
AdamWの基本LR3e-4、betas(.9,.95)、eps1e-8、decay0、全体の勾配clip1、
warmup `min(200, updates//10)`（最低1更新）の後、cosineで3e-5まで下げます。
CSTには明示したfused座標更新Planを使います。条件間でデータ順序を揃え、token予算は
完全な更新単位に切り下げます。developmentは更新0・200更新ごと・最終更新で評価し、
値・勾配の診断は初期・10/100更新・最後に別backwardで行います。runner内で固定した
2つのpromptから64 tokenを生成し、temperature0.8/top-k40、seed7000/7001を使います。

少数固定バッチと1seed・100万tokenのpilotから
始め、挙動とコストを確認後、各実行最大2,000万token・動作する構成で3seedを検証します。
予算拡大や設定変更は別revisionで記録し、失敗runを黙って延長しません。
固定バッチ診断は各条件N256・seed17で、同じ8系列windowを反復して100更新・
延べ102,400 tokenを処理します。confirmation・生成は行わず、未知データの品質結果とは扱いません。

train/validationのCE・perplexityをstep・学習token・実時間の各軸で記録します。
固定promptの生成例、層ごとの値・勾配、FFNと全体のパラメータ数、同期した完全stepの速度と
allocated/reservedピークを残します。診断を計時から分け、compile・準備時間も別記します。
幅の比較はbatch・文脈・計測範囲を揃えます。安定した学習と未知データでの改善、FFNなしより
役立つこと、再現する挙動を有効性の証拠とします。Denseを上回ることは必須にせず、
失敗・不確定な条件も残します。正式な非劣性や一般的なスケーリング則は主張しません。

raw・状態・source/dataハッシュはignored領域か外部に保存します。失敗は残して初期化・
勾配・幾何・コードの原因を分析し、修正を再検証してから広げます。runtime/backendの修正は
torchcst、モデル・データ・研究ツールはこのrepoで管理します。Attention射影の置換は次の研究です。

## 実行入口

この研究はtorchcst commit `9cb6de727ddbf5c2178fe18db6d580c9e181b926`の統合済み
初期化APIを必要とします。repoの既存`[torchcst]` extraは以前のHIGGS依存を固定している
ため、この研究には使わず、隔離環境に対象commitを明示して導入します。
対応するCUDAビルドツール、ninja、tiktoken0.12.0も必要です。

```bash
python -m pip install 'torchcst @ git+https://github.com/takumiecd/torchcst.git@9cb6de727ddbf5c2178fe18db6d580c9e181b926'
PYTHONPATH=src python -m cst_research.experiments.ffn_qualification --cohort small --output evidence/qualification-small
PYTHONPATH=src python -m cst_research.experiments.ffn_qualification --cohort target --output evidence/qualification-target
PYTHONPATH=src python -m cst_research.experiments.ffn_data --output evidence/tokens
```

実receipt・token manifestを確認してSHA256を固定し、`ffn_run --qualification ...
--qualification-sha256 ... --data ... --data-manifest-sha256 ... --kind cst
--width 256 --seed 17 --tokens 1000000 --output evidence/pilot-cst`へ明示して渡します。
pilotの既定wall予算は1時間です。失敗時は部分状態を保存し、自動再開・再試行しません。
metadata検証だけで学習結果を採用したとは扱いません。


## 初期pilot後の半径診断

ownerは半径の被覆を優先し、c8の長時間実験を停止した。半径比較はwidth256、
c8（射影あたり4096 Atom）、seed17、q=0/0.25/0.5を用いる。学習率、データ順序、
予定20M tokenの共通horizonを揃え、まず各条件512 update（524288 token）、
wall上限1200秒で観測する。探索的比較であり、確認実験ではない。
qごとに同じDense目標分散へ二乗平均を合わせて初期化し直す。qを増やすと
振幅や空間相関も変わりうるため、半径だけを独立に変えた介入ではない。
qは振幅依存の上下限の対数間隔における初期割合であり、物理半径そのものでも
学習中の固定値でもない。実際の半径・振幅分布とclipping診断を保存する。
新しい`radius` qualification cohort（各qで両FFN方向、B2 T128、全Atom VJP）
を通してから`ffn_run --q 0.25`または`--q 0.5`を用いる。宣言上の半径上限は50
のままで、実際の半径はそれよりかなり小さくなりうる。準備しただけで結果が
得られたとはしない。この半径仮説を試す間、Atom数とLRへの介入は後回しにする。

CPUの準備確認では、4096 Atom・現在の振幅上限1ではq=1のup射影が目標分散に
届かなかった。これは初期化の拒否であり、学習実験の失敗ではない。q=.5は
二乗平均探索を通ったが、seed17の準備fixtureではup277/4096、down3/4096 Atom
がclipされた。q=.25ではup14/4096、down0である。このfixtureは`make_cst`の
seed17を用い、実際のモデル各射影は名前別seedなので別途reportを確認する。
平均半径up/downはq0で2.49/3.21、q.25で4.35/5.58、q.5で7.14/9.71だった。
これは現在の振幅と半径の結合下での拡大制約であり、Polarやbackendの誤りを
示すものではない。

停止したc8長時間実験の定期checkpointは2800 update・2867200 token、
SHA256 `9e82dd7985d773294c59cfac0f34412a1fe31648fd55a8ed1a5f6ee6a0be6745`。
同じAtom IDの照合で振幅RMSはup .2280→.2359、down .1171→.1237だった。
最終振幅上位10%の平均半径は.914/1.637、下位10%は3.967/3.992で、qはほぼ0。
弱いAtomが広く、強いAtomが狭い状態が共存する。始点と終点から途中の振幅の
累積往復量はわからない。定期checkpointはSIGTERM停止より前のものであり、
未保存updateを復元したとはしない。

## 学習量を揃えた機構の追加確認

半径pilot後はwidth256・seed17でDense、FFNなし、CST q0/q.25/q.5を新しく開始し、
各条件5M tokenを上限として観測する。LRとデータ順序の予定horizonは20M token
で揃え、c8、データ、初期化の目標、optimizer設定を固定する。初期損失が近く
ても3つのCST半径を含め、後の段階で差が出るかを見る。R5や停止したR2を黙って
延長せず、別の追加確認として記録する。約50万tokenだけでスケール実験の
設定を選ばない。

CST診断では`--track-atom-motion`でoptimizer更新ごとに物理振幅を読み、始終点
の差と累積絶対移動量を分けて保存する。診断状態はCPU bufferに保持する。
半径、q、振幅上限近傍の割合は検証checkpoint時に記録する。読み取りはParameter
やoptimizer状態を変更しない。同期したupdate時間には診断を含めず、総経過
時間には含める。診断付きrunを速度・メモリの主比較には使わず、その主張には
別途記録した診断なしの測定を用いる。元のR2/R5証拠は固定したまま保持する。

## 限定した設定の追加確認（R7準備）

R6終了後に証拠を監査してから、幅256・c8・seed17で次の設定を試す。すべての
FFN射影をCSTとし、半径の硬い最小値.5、最大値50、4 block、Dense二乗平均の
目標を固定する。実用設定の確認であり、Atom数の下限探索ではない。

| 設定 | q | 振幅上限 / w_c | Radius birth | 特徴順序 | Atom LR倍率 |
| --- | ---: | --- | ---: | --- | ---: |
| 基準 | 0 | 1 / .3 | 4 | 元の順序 | 1 |
| birthを狭く | 0 | 1 / .3 | 1 | 元の順序 | 1 |
| 較正した広い半径 | 1 | 4 / 1.2 | 4 | 元の順序 | 1 |
| 固定順序の変更 | 0 | 1 / .3 | 4 | 並べ替え | 1 |
| Atom更新を速く | 0 | 1 / .3 | 4 | 元の順序 | 16 |

`radius_birth`は弱い振幅側の下限曲線を変え、硬い最小値を下げる設定ではない。
`permuted`はblockごとに独立したhidden/output channelの固定順序をbufferで保持し、
学習Parameterを増やさない。Denseにも同じ順序オプションを適用し、並べ替えたDense
対照も含める。Denseでは行列に並べ替えを吸収でき、CPUで代数的に確認する。
共分散の並べ替えだけでは、モデルを変更して学習した結果とはしない。

`atom_lr_multiplier`は角度・中心を含むAtom Parameter全体に適用し、通常の重みと
biasは基準LRのままにする。実際のgroup LRはPolar方針のouter timeも制御するため、
振幅LRだけの独立介入ではない。groupの実LR、物理的移動量、半径履歴を記録し、
同じ名目LRから同じ関数更新量を推測しない。

まず各100万token、共通予定20M-tokenのLR・データhorizonで観測し、新しいDense・
FFNなし対照も用意する。探索的な選択にはdevelopment CEを使う。学習前に
`ffn_qualification --cohort recipe`で両256↔1024方向、B2T128、全Atom VJP、
polynomial quadrature、要求したq・境界・順序・LR設定でのoptimizer/model resumeを
確認する。admissionは実Atom数と物理scalar、全モデル設定を照合する。capacity cohort
では係数ごとのoptimizer/model resumeも確認する。独立の生polynomial oracleはbirth4・
max8を仮定せず、軸ごとの宣言scalar境界を読む。準備段階の最初の失敗で、検証fixture
が古い設定を仮定していたことを特定した。この修正でruntime/backendの数学的契約は
変更していない。

GPU gateと限定したタスク更新が実際に通るまではR7は準備である。動作中のR6のソース・
出力を変えない。scalar、特徴順序、LR倍率の変更は新しいqualificationを要求し、
古いreceiptで黙ってadmitしない。動く設定の選択後も幅・縦横比と3 seed確認が必要である。

## 選択した幅・縦横比の数値検証

recipe cohortは`--width 128|256|512`、`--expansion 1|4|16`、
`--coefficient 8|16|32`も受け取ります。実際の両FFN方向をB2T128で検証し、
upはstd.02、downは.02/sqrt(8)を使います。expansion1では同じ正方形でも両射影の
初期化目標が異なることを保持します。他cohortではこれらのflagの非標準値を
黙って無視せず拒否します。選んだq・境界・順序・Atom LRは数値検証と学習で揃えます。
別shapeやAtom数のreceiptで学習をadmitしません。

たとえば設定を選択し、明示的な2048対応operatorソースを検証した後、幅512では
`ffn_qualification --cohort recipe --width 512 --expansion 4 --coefficient 8`
に選択した設定のflagを付けます。長い学習前に、実際の4 blockでのタスク更新と
checkpoint監査も行います。このflagでbackend対応を拡張するわけではなく、未対応の
組合せは検証に失敗します。入口の準備だけで幅・縦横比の学習や新Atom数の検証済みを
意味しません。
