# 独立した半径の損失勾配学習：RTX 6000 Ada FFN予備比較

[English（正本）](ffn-independent-radius-20261108.md) | 日本語

既存TinyStories FFNに対して物理半径を直接学習すると改善するかを、実行前に
条件を固定して1seedで調べる。最適半径、収束、非劣性、Atom数の下限は求めない。

## 固定する比較

既存モデル・学習器を使う。幅512、pre-norm 4ブロック、8head、context128、
FFN拡張4倍、embeddingと語彙出力は共有Dense。AttentionもDenseのまま。
FFNの8射影をCSTにし、各射影は `A=8 sqrt(d_in d_out)=8192` Atom、
M8 cell-mass、base tile1/merge2、周期境界の単位格子とする。順番を固定する。

1. `polar`：従来のrational-zero-opening、minimum scale0.001、
   radius birth0.5、q0、振幅上限1。
2. `learned`：同じPolar初期値を物理値 `(a,r,mu_out,mu_in)` に変換し、
   4成分を通常のAdamWで更新。
3. `fixed`：learnedと完全に同じ物理パラメータとoptimizerを使い、
   半径勾配だけをglobal clipの前に0にする。半径のmomentも0を保つ。

変換では復号した振幅・半径を一度FP32に丸める。座標と通常のモデルパラメータは
完全に共有し、learned/fixedの初期tensorは完全一致させる。Polarと物理表現の
初期logitは事前固定したFP32許容差以内で一致させる。Polarとの比較には振幅の
更新座標変更も含まれる。**物理表現内のlearned対fixedが半径学習の切り分け**。

独立半径はforward内でclampしない。AdamW更新後に半径を `[0.5,50]`、
振幅を `[-1,1]` に射影する。座標は演算子内で周期的に扱う。この境界は今回の
実験条件であり、無制約の最適解ではない。AdamWはbase LR0.0003、
全Atom成分LR16倍、betas(0.9,0.95)、decay0、global clip1、FP32・TF32無効。
結果を見て学習率を選び直さない。

seed17、batch2×context128×蓄積4、976更新・999424token。
20M-tokenのschedule horizonと学習prefixを変更しない。token manifestは
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。
8192-tokenのDEVのみ評価し、confirmationは使わない。
Denseは再実行せず、同じ設定の過去結果を参考として区別する。

## 実行前検査と観測

実行前に完全なcommitと各ファイルのhashを固定する。予約済みRTX6000 Ada
48GBの実機・runtimeを確認し、既存の永続controllerで直列実行する。
明示指定の非公開physical LM Planを使い、入出力両側の半径微分を含む既存native
物理微分を再利用する。公開defaultを変えず、半径微分を省くPolar reductionは使わない。

学習前にCUDA回帰検査、独立Gauss求積による物理微分、両FFN形状の全8192Atom・
256非連続行のCPU参照VJP、同じgraphの再backward、モデル全体の初期値一致、
learned半径勾配の有限・非ゼロ、fixedのmask、各条件の完全更新とcheckpoint・
optimizer復元を検査する。

step0、1〜5、20ごと、976で55回観測する。8層それぞれの振幅・半径の
平均/RMS/p10/p50/p90、境界占有率を残す。各更新後に同じ固定probeで、clip前の
損失勾配を更新タイマー外で測る。Polarに損失の半径勾配はなく、fixedはmaskで0。
これは固定probe勾配であり、蓄積平均した学習勾配ではない。
境界占有率と境界に当たった更新提案数は区別する。

学習曲線を更新数・token数に対して重ね、半径推移と半径勾配RMSも重ねる。
同期した完全更新時間、準備時間、allocated/reserved最大メモリを残す。
診断時間はwall timeに含み、更新時間には含めない。速度解釈では固定実行順を明記。
学習後はcheckpointを独立復元し、DEV再計算、学習後VJP、fixedの半径/moment、
次の1更新とoptimizer stateの完全一致を確認する。証拠を回収しhash確認するまで
COMPLETEDにしない。checkpoint・データ・source archive・native buildはclusterに
保存する。失敗した凍結実行を編集して再開しない。

runnerは `cst_research.experiments.ffn_radius_qualification` と既存
`cst_research.experiments.ffn_run --kind cst --attention-kind dense
--linear-backend levels --radius-mode {polar,learned,fixed} --track-radius
--development-only`。上記の固定条件とhash付き検査receiptを渡す。

実行と結果の識別情報はsource凍結後に別途記録する。
