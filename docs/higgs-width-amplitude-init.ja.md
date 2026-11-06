# 幅を考慮したamplitude初期化

[English (canonical)](higgs-width-amplitude-init.md) | 日本語

限定した実行・独立した全保存証拠の検証は完了しています。[結果と限界](higgs-initialization-results.ja.md)を参照してください。以下のprotocolは元の設計を保持し、準備中・将来形の記述は当時の記録です。

指定されたamplitude標準偏差`0.1/sqrt(A/N)`を、従来の`0.1/sqrt(A)`、
および以前測定した固定gainと比較する独立の探索実験です。未実行の
He・自己エネルギー方式の試作は使用しません。

幅1024・隠れ8層、内部のbiasなしCST7層には各4096 Atomsを用い、通常の
Dense入出力も学習します。各GPUで、従来Dense・旧初期化CST・新初期化CST・
旧初期化CST＋固定gainの4条件を新規実行します。seed17、従来の通常層初期化、
FP32・TF32無効、eager・実batch4、stockq1 AdamW/CSTAdamW、学習率0.0003、
同じ保存済みseed117順序に揃えます。各条件1 epoch・25,000更新とし、
最終状態を固定選択します。HPO・早期停止・DEV最良選択はしません。
RTX4090とA100MIG3g.40gbは、厳密なruntime pinを持つ別ハードウェア群です。

## 変更内容

変更しないpublic975 factoryから同じ正規乱数と座標を得ます。新条件では
初期polar第1成分をsqrt(N)倍し、従来と同じ対称amplitude比クリップを適用し、
第2成分をsqrt(1-ratio²)で再構成します。N1024では32倍がFP32で厳密であり、
正規分布の標準偏差は0.0015625から0.05になります。alphaは通常のFP32符号化誤差を
除いて同じゼロ初期化です。座標と通常層の全Tensorは保持します。
amplitudeに伴うradiusの変化は意図したもので、形状を保存する介入ではありません。

新旧の直接初期化条件はいずれもgain1・正規化なしです。gain対照は以前と同じく、
TRAIN先頭256件の特徴から、同じ初期DenseのReLU前RMSに逐次合わせます。
ラベル・DEVは使用しません。後から試作したデータ非依存He gainとは異なります。
公開runtime・演算・微分・optimizerは変更しません。

runtimeの構造契約名は既存のcase名を保持しますが、各checkpoint・reportの実験identityに
実際の条件名と初期化方針を別途記録します。新しい初期値も特別な復元処理を使わず、
変更しないnative model/optimizer loadによって保存済みの値から復元します。

## 測定・検証

全4条件について、新規の全ネットワークY・dX・全パラメータのFP64参照VJP、
変更しない数値gate、独立31-Atom求積、scalarAdam、通常8更新対4＋load＋4、
観測付き初回＋通常7更新対通常8更新の全状態一致を確認してから学習します。
増幅cotangentは診断用であり更新には使用しません。
合成検証80更新と実データ100,000更新は区別します。

0・1000・5000・10000・15000・20000・25000更新でTRAIN100kとDEV50kを評価します。
各時点で独立にloadしたコピーに同じ最初のTRAIN batch4を入力し、全層の生の
順伝播値・逆伝播勾配とパラメータ勾配を保存します。probeではoptimizer更新を行わず、
元の全coreが変化していないことを検証します。実際の初回学習更新はさらに生勾配・
採用Parameters・moments・clock・変位を保持します。採用状態からのAdam分母は
報告用の再構成であり、native中間bufferの直接取得ではありません。

同期した通常更新区間の時間とpeak allocated/reservedを記録し、probe・評価・
校正・IO費用と分けます。raw6GiB・JSON64MiBを上限に、source・data・runtimeの
hash、最終checkpoint、全予測配列を含む完全な証拠をserverに保存します。
主実験confirmation・将来confirmation・公式testは採点しません。
1 seed・1 epochから非劣性・収束・最適な初期化・Atom数の法則は主張しません。

entrypointは`python -m cst_research.experiments.higgs_width_amplitude_run`です。
`--help`に不変consumer・外部source・入力の明示pathとGPU選択を示します。
GPU実行前にconsumer commit・manifest・入力を固定します。
