# DenseのRMSに合わせる生のamplitude初期化

[English (canonical)](higgs-rms-amplitude-init.md) | 日本語

限定した実行・独立した全保存証拠の検証は完了しています。[結果と限界](higgs-initialization-results.ja.md)を参照してください。以下のprotocolは元の設計を保持し、準備中・将来形の記述は当時の記録です。

Atomの初期amplitudeを調整して初期DenseのReLU前RMSに合わせます。
実行時gainは1・正規化はNONEです。独立の探索的な初期化条件であり、
公開defaultは変更しません。以前の係数0.1固定の幅対応実験は元sourceで固定します。

N1024・隠れ8層・CST7層・各4096 Atoms・seed17、従来の通常入出力初期化、
TRAIN100k・DEV50k、実batch4・FP32・TF32無効・stockq1・学習率0.0003、
1 epochの最終固定選択を同じにします。同じGPUで検証済みのDense・旧初期化・
幅対応係数0.1・旧初期化＋gainと比較します。別jobの時間は記述的であり、
1 seed・1 epochから収束や非劣性を主張しません。

実データ学習前にTRAIN先頭256件の特徴だけを使います。内部層を順番に処理し、
元の座標・標準化された正規乱数を保持したまま、クリップ前amplitude標準偏差が
`k/sqrt(A/N)`となる層ごとの係数kを探します。従来の対称比クリップとゼロalphaの
polar符号化を使います。amplitudeによるradius変化を許容し、クリップ数を記録します。
同じ初期Denseの256件に対するReLU前RMSを目標とし、その後のReLUには実際の
校正済みCST出力を使います。ReLU後RMSも記録しますが一致条件にはしません。
ラベル・DEV・heldoutは初期化に使いません。

係数0・.05・.1・.2・.4・.8・1.6・3.2を順に評価して符号の異なる区間を探し、
最大32回二分探索します。実測の相対RMS誤差0.5%以内だけを採用し、全体の単調性は
仮定しません。到達不能や不連続で一致しない場合は探索履歴を残して失敗とし、
gainへのfallback・クリップ規則変更・runtime上限拡大は行いません。
実際のFP32候補Atomsをpublic CPU FP64参照で評価します。256件をB4ずつ測った
native FP32 witnessでもDenseとのRMS誤差1%以内を確認します。係数・生の初期Atoms・
クリップ数・radius診断・目標/参照/native配列・途中経過を保存します。

初期化終了後は校正手続きを固定し、Atom Parametersを通常どおり学習します。
学習中にRMSを継続的に拘束しません。新規の全ネットワークVJP・独立求積・
scalarAdam・通常更新/復元・観測の一致を確認してから学習します。
合成ネットワーク20更新と別のscalar8更新は実データ25,000更新に含めません。
新runnerは数値的な学習・検証フローに明示的なmodel factoryを渡し、既存の
default factoryの動作を保持します。

0・1000・5000・10000・15000・20000・25000更新で全TRAIN/DEV予測を保存し、
最初のB4固定入力を独立copyに与えて全層・全パラメータの勾配を観測します。
probeではoptimizer更新を行わず、元の全coreを変更しません。実際の初回更新の
勾配・moments・採用変位も保存します。完全な証拠をremoteに保持し、独立の
保存配列/状態検証を終えてから結果を受け入れます。
主実験・将来confirmation・公式testは採点しません。

entrypointは`python -m cst_research.experiments.higgs_rms_amplitude_run --help`です。
この限定した追加実験は確保済みRTX4090で行い、元のA100幅対応初期化比較は
変更しないsourceのまま継続します。
