# HIGGS: Static・RigL-style参照の学習率調整と比較

[English (canonical)](higgs-dst-comparison.md) | 日本語

## 状態と実行中の研究との関係

これは新たに承認された比較プロトコルであり、完了結果ではありません。
既存のseed17 DST pilotは今回の追加を促した探索的development証拠です。
新しいheld-out結果や、DSTを調整済みのbaselineとは扱いません。

元のCST研究はtorchcst commit `97580582ed28c4e494bf5c2656bbe94a85c544c3`に固定し、
元の8構成の非劣性familyを変更しません。追加DSTと比較controlの設定・checkpointが
全て固定されるまで、そのconfirmation評価を保留します。
confirmation後半の結果をLR・学習期間・checkpoint選択に使いません。公式testも開きません。

研究consumerは外部torchcst commit `f37729e61dc11a57e76103aa2457c2be11c4c306`
（[PR #143](https://github.com/takumiecd/torchcst/pull/143)）のDST実装を利用します。
runtimeを複製せず、privateな実験用CUDA Graph adapterを公開APIとして扱いません。
最初の実装範囲はqualificationとtuningです。対応controlと比較全体のfreezeがそろうまで
confirmation・assessmentは拒否しなければなりません。この計画の記録は、後続stageの
実装完了・実行済みを意味しません。

## consumerと交換境界

consumer入口は`cst_research.experiments.higgs_dst_comparison`、pool wrapperは
`src/cst_research/experiments/higgs_dst_comparison_driver.py`です。wrapperは固定consumer
commitとstage、幅・方式・divisor等を受け取ります。承認済みの共有pool経路を使い、
これらのmodule自体はGPUを割り当てません。最初のdeliveryは`qualify`と`tune`を実装します。
`freeze-width`では元の期間の検証済みimportと共通期間の準備状況を示せますが、
controlの全workflowを実装・reviewするまで最終比較freezeの公開を拒否します。
このdeliveryの`confirm`と`assess`は未提供で、呼び出しても拒否します。

外部source archive/manifest、prepared data/data-manifest、consumer-source manifestを
独立に供給してrunへ結び付けます。source manifestは完全commitとmember hashを持ち、
recoveryにはarchiveとreceipt SHAを要求します。controllerへ外部source・data・output・
state・identityの所在を渡し、任意のoriginal-selection indexで元研究のartifactを参照します。
元の実際の`frozen-width.json`をfile SHAで固定してimportし、identity・width・horizon・
全5構成とLR/candidate一覧を検証します。手で転記したLRを代わりに使ったり、
曲線から固定済み期間を推測したりしません。元の確認seedとdevelopment-only選択flagも維持します。

```bash
python -m cst_research.experiments.higgs_dst_comparison --help
```

## 比較方式・データ・予算

全モデルは28→N→N→2、ReLU×2、入力・出力はdense、中間bias無し、全層学習です。
N={1024,2048}、divisor={1000,500,200,100}、α={0.001,0.002,0.005,0.01}。
各幅・密度でstatic maskと固定したrow-balanced RigL-styleを比較し、DSTは合計16構成です。

$$
A=\lfloor N^2/\mathrm{divisor}\rfloor,\qquad K=4A,\qquad
P_{\mathrm{active}}=K+31N+2.
$$

Kは中間層のactive scalar weight **slot**数で、新生したゼロ値slotも数えます。
CSTのatom座標scalar数4Aと合わせるものであり、保存Tensorの全要素数・非ゼロ数・
mask/indexのbytesではありません。StaticとRigLは同じmask・値から開始します。
行ごとのbudget差は最大1で、再配線後も各行budgetを維持します。
固定sourceの行fan-in初期化、通常層の初期値、seedごとのデータ順を保ちます。
mask探索とCSTの幾何は異なる制約を持ち、active slotが同数でも関数クラスの同一性は示しません。

prepared archiveのSHA256は共通で、
`62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac`です。
学習は公式trainingの先頭100,000行。標準化は公式training10,400,000行のみでfitしたものを使います。
公式validation内[0,50000)をdevelopment、[50000,100000)をconfirmationとします。
indexはzero-based半開区間で、公式testは評価しません。

真batch4、FP32、TF32 off、CPU math1 threadを維持します。固定AdamW契約は
LRを調整し、betas=(0.9,0.999)、eps=1e-8、DST重みと通常層のWD=0.01です。
dense masked referenceは固定sourceのoptimizer実装を維持し、方式間の算術が
byte同一だとは主張しません。

RigL-styleはcadence250、初期drop fraction0.3、全minibatch予算の0.75までのcosine減衰、
row-balanced growth、列indexによるstable tieを保ちます。eventは正のminibatch stepのうち
250の倍数で、floor(0.75×全minibatch数)より小さい時点に限ります。global重み絶対値thresholdで
pruneし、各行に少なくとも1つ生存slotを残してから初期行budgetまでgrowします。topology minibatchでは
全層AdamW更新を置換します。生存した重み・momentは保持し、prune/growされた全座標の
重み・momentは、直後の同座標再生も含めてゼロにします。topology minibatchでglobal Adam
clockを進めません。minibatch・処理example数と実optimizer更新数の両方を記録します。
等予算はexample提示数を合わせる意味で、Adam更新数を同じにはしません。
20epoch runは初期値から新規に走らせます。10epoch checkpointから延長すると、
最終期間に依存した本来のrewiring scheduleが変わってしまいます。

## 公平なLR探索と共通の比較期間

development seedは17です。16個のDST構成ごとにLR={0.0001,0.0003,0.001,0.003}を
全て10epoch学習し、各epochでdevelopment全50,000例を評価します。
checkpointはdevelopment cross-entropy最小、完全tieなら早いepochを選びます。
LRはbest-development cross-entropy最小、完全tieなら小さいLRを選びます。

最良base LRが端なら、その方向へ1候補だけ追加します。
下端0.0001なら0.00003、上端0.003なら0.01で、追加点も10epochです。
最終選択が評価範囲の端ならboundary flagを残します。
失敗を除外したり、性能の負けを再試行で置き換えたりしません。

幅Nごとに、dense＋4CST構成で決まった元の固定期間T_original(N)を参照します。
その幅の全8DST構成の選択済み10epoch候補の**いずれか**が
best_epoch≥9かつCE(epoch8)−CE(epoch10)>0.001ならDST延長triggerを満たします。

$$
T_{\mathrm{DST}}(N)=\begin{cases}20&\text{if the DST trigger fires},\\10&\text{otherwise},\end{cases}
\qquad T_{\mathrm{comparison}}(N)=\max\bigl(T_{\mathrm{original}}(N),T_{\mathrm{DST}}(N)\bigr).
$$

比較期間が20なら、その幅の**全DST構成**で10epoch時の上位2LRを初期値から20epoch再実行し、
同じdevelopment規則で選択します。元研究だけが20を要求した場合も同様です。
同じ幅のStatic・RigL・dense・CSTの比較は全てこの共通期間を使います。
best_epoch≥19かつCE(epoch18)−CE(epoch20)>0.001なら改善が続く末尾として記録し、
confirmationを見て予算を暗黙に延長しません。

controlの扱いを分けます。

| 元の期間 / 比較期間 | dense＋4CST比較control |
|---|---|
| 10 / 10または20 / 20 | source/data/configuration/hashを検証してから、元の固定設定と3seed checkpointを再利用する。 |
| 10 / 20 | 別の比較namespaceで、元のdevelopment候補一覧の上位2LRを各構成20epoch、変更しない`97580582...` cell runnerで再実行する。そこで選択し、5構成全てを3確認seedで新規学習する。 |

後者は該当する幅ごとにcontrol LR10run＋新規control15モデルを追加します。
元研究の期間・選択記録・checkpoint・8構成の主解析は変更しません。
追加controlを完了できなければ同期間比較は未準備です。
10epoch controlと20epoch DSTを混ぜ、公平な確認だと主張しません。

## confirmationを開く前のfreeze

paired確認seedは29,43,71で、初期値から学習します。DSTは16×3=48個のbest-development
checkpointを持ちます。**いかなる**confirmation後半のevaluatorより先に、48個全ての
checkpoint/data/configuration identityと、dense/CST比較control30個全てのidentityまたは
検証済み再利用参照をfreezeします。元研究も元の30checkpoint manifestを固定してから
assessmentへ進みます。HPO未完了・control来歴未確定・freeze不完全なら評価を止めます。
各seedのcheckpoint選択はdevelopmentだけで行います。

confirmation後半をheld-outと呼べるのは、過去のpilotやassessmentでも開いていない場合だけです。
この前提を明示記録します。既に見ていれば正式評価前に停止して計画を改訂し、
新しいheld-out confirmationと言い換えません。

## 統計上の主張

16個の（幅・密度・DST方式）構成ごとに、同期間dense controlとの差をseedごとにpaired比較します。
accuracyとROC-AUCの許容差は両方0.002で、accuracy0.002は0.2ポイントです。
同じdf=2の片側paired-seed t検定とco-primary IUTを使います。

$$
p_{\mathrm{configuration}}=\max(p_{\mathrm{accuracy}},p_{\mathrm{AUC}}),
\qquad16p_{\mathrm{configuration}}\le0.05.
$$

seed標準誤差がゼロならmean_difference+margin>0のときだけp=0、それ以外は境界も含めp=1です。
非劣性を支持・悪化の強い証拠・判定保留を区別します。悪化診断にtwo-sided同時seed-t bandを
使うなら、新familyの32endpointにBonferroni補正し、主IUTから分けます。
pointwise区間とpaired-example bootstrap区間はそれぞれの範囲を持つ診断です。
3seedは検出力が低く、paired-seed差の正規IID近似も検証できません。
seedとexample両方へのjoint confidence保証はしません。

元CSTの8構成familyと新DSTの16構成familyは、それぞれfamilywise error0.05です。
全24主張へjoint95%を保証するものではありません。CST対Static/RigLの直接比較と
learning curve差は、confirmation前に別の固定推論familyを登録しない限り記述的に扱います。
最小のsupported tested densityだけを報告し、小さい点の欠落・判定保留を残します。
単調性や普遍的な最小値は仮定しません。

## 資格確認・曲線・解釈

実学習の前に固定外部source/data identity、masked CPU FP64 Y/dX/dW参照、
両幅・密度両端でStatic/RigLのeager/Graph軌跡を検証します。
実topology budget、全state/gradient、dense growth gradient、moment reset、clock、event数を
確認し、失敗とqualification証拠を残します。guardを迂回したり、公開capture APIを拡大したりしません。

選択済みrunだけでなく、全HPO・延長期間・確認用学習runの曲線を保持します。
訓練損失とdevelopment CE/accuracy/AUCをminibatch・処理example数・累積training時間で描き、
実Adam更新数、準備・評価費用と、取得できるend-to-end walltimeも記録します。
HPO seed17と独立確認学習seedは図でも区別します。曲線はdevelopmentの指標であり、
confirmationを繰り返し評価するものではありません。

この参照実装はdense重み・moment・maskを保持し、dense growth gradientを計算します。
時間にはtopology処理とevent監査hashが入ります。同期完全step・準備込み時間・allocatorの
allocated/reserved peakを残しますが、疎kernel DST実装の速度・メモリ効率は主張しません。
別VMのHPO時間は探索的で、同一job比較controlのjob/hardware/runtime来歴を保持します。

raw source/result archive、候補一覧、mask、checkpoint、topology hash、curveはignored外部証拠へ
保存します。registryには相対locatorだけを置き、dataset・認証情報・絶対ユーザーパスは含めません。

## Stage2の評価境界と解析

Stage1はcommit `22f2810b1bc51a599e51c8936acac310ca9455ef`の資格確認・tuning consumerに
固定します。別のStage2 consumerでcontrolのimport/再調整、独立学習、global freezeを追加します。
`cst_research.experiments.higgs_dst_assessment`はCPU admission、1回限りの評価schedule、
CPU統計を提供します。これらは合成CPU検査で確認しており、この記述はGPU workerの実行済みを
意味しません。

比較側の`original-confirmation-freeze-proof.json`は、検証済みの元学習jobと幅別freezeから
組み立てた、変更しない元`97580582...` assessorが生成する30モデルpayloadそのものです。
raw fileとcanonical JSONのSHA256でglobal78 freezeへ結び付け、比較側のproofと明示します。
その生成のために元assessorを呼んではいけません。元のコマンドはfreeze生成直後に後半を評価します。

順序を次に固定します。

1. 元研究と比較研究の学習・選択を全て完了し、78モデルmanifestと元30モデルproofを封印する。新assessorの`audit`は全model/data/sourceを検査するが、benchmarkをimportせずconfirmation例も読まない。
2. global admission後、変更しない元975の`assess`を1回だけ実行する。実際の30モデルfreezeのcanonical内容はproofと一致しなければならない。
3. 新`evaluate`へ`--original-assessment-job`を渡す。元jobのsource/result archive・receipt・manifest・freeze・保存予測metricを検証し、再利用controlには既に1回評価した予測をimportする。そのモデルを再評価しない。ここで評価するのは新DST48モデルと、必要なら別途20epochで学習した新controlだけ。
4. 完成した保存assessment indexをCPU `analyze`で解析する。元の8構成解析は独立したまま変更しない。

controlを再利用する場合、元assessment bundleが欠けていれば新たな評価も止めます。
選択checkpointの欠落・改変、共通期間・source manifest・producer/再利用参照の不一致、
比較family未完了でもadmissionを拒否します。新workerはconfirmationを読む前に全体のgateと
自分の未実行attempt markerを再確認します。import先は2つの固定commitについてbyte検証した
外部source directoryだけで、result archive内のコードは実行しません。公式testも読みません。

freezeにはDST48個と同期間control30個のidentity、固定checkpoint/metadata/learningファイル名、
source-manifest digest、dataset identity、幅別期間、元30モデルproofを封印します。
`--frozen-index-sha256`はその不変artifactと一致しなければなりません。
`--source-index`は固定commitから検証済みcache directoryと外部manifestへの対応表であり、
任意のsource pathを実行する許可にはしません。

予測・metric・file/array hash・評価来歴をmodel identityとfreeze SHAに結び付けてcacheします。
完了結果は再開時に検証・コピーし、再評価しません。開始済みで未完了のattemptは保全して
拒否し、自動retryはしません。明示的に復旧してから継続します。
元のevaluation metadataもimportした予測と一緒に保持します。

```bash
python -m cst_research.experiments.higgs_dst_assessment --help
```

stageごとのoptionは次のとおりです。

| Stage | 入力 |
|---|---|
| `audit` / `evaluate` | `--frozen-index`, `--frozen-index-sha256`, `--campaign-state`, `--data`, `--source-index`, `--output` |
| control再利用時の`evaluate` | 加えて`--original-assessment-job`で検証済み元975のpool job全体を指定 |
| `analyze` | `--assessment-index`, `--output`。保存予測だけを読む |

評価は承認済みjob driver・resource手順で行い、このmoduleからGPUを割り当てません。
CPUの予測検証・解析には記録した実験環境のNumPyを使い、registry CLIは標準ライブラリだけのままです。

解析は固定16構成IUTと、それから分けた32endpoint同時悪化診断を実装します。
CST対DSTの直接比較にはseed別差・平均・標準偏差/誤差と、**多重補正しないpointwiseの
両側95% seed-t区間**（df2）を出し、固定例・seed正規近似に条件付く記述的な値と明示します。
多重補正したCST対DST優位は主張しません。この実装はpaired-example bootstrap区間を計算せず、
未実施を出力に明記してseed/example joint coverageを示唆しません。
