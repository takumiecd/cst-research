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
最適な式や最小係数は求めません。並列に使うA100 MIG40GB条件では、その環境での数値・
runtime検証、厳密なGPU確認、対応する3条件の実行が必要です。速度・メモリはデバイス別に
記録し、異なるGPUを混ぜて幅スケーリング曲線にはしません。

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
