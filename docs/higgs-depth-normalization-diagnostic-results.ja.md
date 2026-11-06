# HIGGS：最初の更新における正規化診断の限定結果

[English](higgs-depth-normalization-diagnostic-results.md) | 日本語

**9ケース・各8更新の診断が完了し、保存証拠が独立CPU admissionを通過しました。**
最初の実B4更新ではA104 stockの座標勾配は非ゼロでも極小で、物理座標は動かず、
固定gain／正規化を入れた場合は座標RMS変位が約1e−4でした。これは初期1更新の
証拠であり、学習曲線全体の原因説明や正規化で品質差が埋まる証明ではありません。
development／heldout評価、HPO、長期学習、最小α／スケール則、throughput比較、
既定値への採用は行っていません。以前の負の結果・部分的改善も維持します。

実行したconsumerは、この結果ノートのbranchではなく
**4f777a5f1ef38cca7031cf9c637d0fa6fc0d5e7b**です。85ファイルとexternal975の
1188ファイルを検証しました。公開runtimeは
97580582ed28c4e494bf5c2656bbe94a85c544c3のままです。
[科学的プロトコル](higgs-depth-normalization-diagnostic.ja.md)、
[以前の限定結果](bounded-validation-results.ja.md)、
[機械可読の導出要約](higgs-depth-normalization-diagnostic-summary.json)も参照してください。

## 実行とadmissionの範囲

root所有のRTX4090 foreground実行1回が正常終了し、固定9ケースで**真のtask更新を
各8回**（合計72回）行いました。B4／FP32／TF32 off／eager、seed17／LR.0003です。
28→1024→1024→1024→1024→2、biasなし正方中間層3つ、学習するDense IO、stock q1
CSTAdamWを使います。affineなしRMS／LNのεは共通1e−5、stock Adamのεは1e−8で、
β／decayの意味と全数値gateを維持しました。

GPUはNVIDIA GeForce RTX4090、CC8.9、128SM、25262096384 bytes、承認profileは
`gpu4090-tf2-18-0-pytorch2-6-0-cu126-keras3-8-0`です。実際の隔離runtimeは
Python3.12.9、Torch2.11.0+cu126、CUDA12.6、Triton3.6.0、NumPy2.0.2、
driver570.133.20です。runtime pinのSHAは
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`です。

保存した全151ファイル（raw2,643,520,928 bytes）のSHAをadmitしてから意味を検査しました。
全9ケースの通常Y/dX/全勾配、norm／packed列／material gate、8対4+resume+4状態、
有限性／所有権／clock、task状態0/1/2/4/8が成功しました。producerは合成入力の各stepで
観測あり／なしnativeの一致を確認しましたが、独立した保存raw比較は**最終8**であり、
全stepのraw履歴を独立再確認したとは扱いません。task入力／ラベル／順序とforward／
backward要約も照合しました。native scalar Adam再構成の最大相対誤差は9.7044e−8です。
合成qualificationはtask72更新とは別で、probe前に初期化し直します。Denseを含む各caseの
増幅componentwise失敗も保存したままです。事前に決めた必須の通常／norm／material
基準は成功していますが、任意の増幅componentwise cotangentを保証するものではありません。

## 256行の校正probeではなく、最初の実task batch

表示したforwardは、共通のseed117初epoch順序の先頭**4例**を使う、**更新1より前**の
値です。同じbatchでCE／backwardと次の更新を行います。固定gainは別のTRAIN特徴256行
だけを使い、ラベルなしで校正しました。そのprobeのRMS一致は、このB4での一致を保証しません。
信頼区間や全データの品質スコアはありません。

\[
\rho=\frac{\sum_i\max(z_i,0)^2}{\sum_i z_i^2}
=\left(\frac{\operatorname{RMS}(\operatorname{ReLU}(z))}{\operatorname{RMS}(z)}\right)^2.
\]

| Case | Last pre-ReLU RMS | Last post-ReLU RMS | Positive energy retained (%) |
|---|---:|---:|---:|
| dense-stock | 0.0362681 | 0.026206 | 52.2099 |
| dense-rms | 0.0402553 | 0.0290711 | 52.1527 |
| dense-ln | 0.0402539 | 0.0285256 | 50.2174 |
| cst104-stock | 2.56882e-10 | 7.71244e-12 | 0.0901398 |
| cst104-gain | 0.0327717 | 0.000983915 | 0.0901398 |
| cst104-postgain | 0.650299 | 0.0195241 | 0.0901398 |
| cst104-rms | 0.0330892 | 0.00100207 | 0.0917109 |
| cst104-ln | 0.0324452 | 0.00554607 | 2.92193 |
| cst1048-stock | 4.81402e-10 | 2.74249e-10 | 32.4544 |

A104 stock、ReLU前目標gain、ReLU後目標gainの最終ReLUエネルギー保持率は約0.09014%です。
正の定数gainは理想実数では符号／エネルギー分割を維持します。ReLU後目標の校正では
このbatchの後段RMSは.0195241となり、前目標の.000983915、Denseの.0262060と異なります。
校正probe以外でDenseと厳密に一致する必要はありません。保持率はRMSで0.09171%、
LNで2.92193%ですが、Denseの52.2099%より低い値です。これはエネルギー分布であり、
activationの平均、永久に死んだneuron数、最終精度ではありません。

## 実座標更新とεの集計範囲

各CSTケースの範囲は、3層のμ_output／μ_inputという**6 field block**に対するものです。
勾配と変位RMSはそれぞれのfieldのatom全体で計算し、変位はN1024物理座標単位での
最小符号付きtorus変位です。ε支配はsqrt(vhat)<εで、native proposalの実m/v、clock、
param-group設定から再構成し、独立scalar照合を通しています。ここでの割合は**準備後
勾配がゼロの座標を除外**し、勾配エネルギーではなく成分数を数えます。機械可読要約には
ゼロ勾配を含む全成分の割合も残しています。

| CST case | Coordinate prepared-gradient RMS | Nonzero gradient (%) | ε-dominant among nonzero gradients (%) | Accepted physical μ-displacement RMS |
|---|---:|---:|---:|---:|
| cst104-stock | 2.83e-14–4.27e-14 | 60.6–73.1 | 100–100 | 0–0 |
| cst104-gain | 3.61e-06–5.46e-06 | 58.7–75 | 79–86.9 | 7.98e-05–0.000104 |
| cst104-postgain | 7.19e-05–0.000109 | 62.5–77.9 | 76.9–81.5 | 9.72e-05–0.00012 |
| cst104-rms | 3.66e-06–5.56e-06 | 86.5–100 | 78.8–88.5 | 9.29e-05–0.000128 |
| cst104-ln | 8.82e-06–0.000125 | 65.4–100 | 0–65.4 | 0.000164–0.000304 |
| cst1048-stock | 7.94e-13–1.27e-12 | 99.6–100 | 100–100 | 0–7.36e-09 |

最初のA104 stock更新では座標6 fieldすべてで物理変位がゼロでした。encoded値には
0↔1024の変更が1つありますが、torus上では同じ点です。補正によって測定した勾配／受理
変位が増えるため、スケールから最初の更新への狭い関係は支持されます。しかし以前の
3 epoch停滞の原因が確定したわけではありません。特にA1048 stockも初回の非ゼロ座標
勾配ではε支配率100%なのに、**別のRTX3090実験であるf6の3 epoch cohort**では
学習しています。これはdeviceをまたぐ統制された因果証明ではなく、初期ε支配率だけでは
**A104の停滞を説明できません**。FP32座標の分解能、勾配分布、その後のmoment、
gate変化、学習時間の影響をこの最初のbatchだけでは解決できません。

## 関数写像と、維持した報告上の限界

固定正gain、biasなし正方層とReLU、制約のない最終Denseについては、既に示した
理想実数の再parameter化が使えます。

\[
F_g(\theta)=F_{stock}(T_g\theta),\qquad
W'_{out}=\Bigl(\prod_{\ell=1}^{3}g_\ell\Bigr)W_{out},\quad b'_{out}=b_{out}.
\]

RMS／LNは入力依存の正規化を挿入し、LNは中心化も行います。同じA・同じ初期学習
テンソルでも、正規化variantの関数族が同じだとは示せません。gainと同様の同値性は
証明しておらず、最初の更新差を容量を統制した品質比較とは扱いません。activationの
スケールを合わせるだけで、atom数が一意に決まるわけでもありません。

次の報告上の限界を維持し、測定したsourceは変更していません。

- 凍結した変位reporterは+N/2を使うwrapで、極小の範囲内変位を丸め落とすことが
  あります。別途、安定な最小torus変位の導出を実delta37,632件についてlibmのIEEE
  remainderと照合しました。legacyとの差は7件で最大3.6374e−14です。保存step8の
  A104 stockには実際に9.5407e−15の座標deltaがあり、**8更新すべてで文字通り完全ゼロ
  だったわけではありません**。legacy／修正版の報告、生値、凍結4fをすべて維持しています。
  これは導出reportの訂正であり、科学的再実行や許容差変更ではありません。
- CPU reader01は非Tensor `_extra_state` の比較で停止しました。reader02がmetadataと
  tensorを再帰的・厳密に確認し、同じraw attemptが成功しました。失敗したreaderと全ての
  増幅componentwise失敗も保存しており、別のGPU attemptや推論を代入していません。

## 再利用できる来歴

consumer manifest:
`aa942f52f88d9a26a950eb51203df948216bbb50fb437c410b9dc8d9b86bbd53`。
科学的protocol:
`0a267cc7fe634426ada6a515bdddf3459b7f7fa2d34797b84d38f65860cc3d33`。
データ／露出境界はf6と同じで、学習だけでfitしたartifactのSHAは
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`です。
公式testや元のHIGGS confirmation payloadは評価していません。

外部locator:
`$CST_RESEARCH_EVIDENCE_ROOT/higgs-depth-normalization-diagnostic-20261106/evidence-index.json`。
このignored indexが元の保存report、source-bound PNG／PDF／CSV、receiptを結びます。
raw checkpoint／data／log archiveはこのノートへcommitしません。結果gzip全体のSHAは
`882fa8a3e96b9214793fa4bf68f0c203de8bab83e98b4750d0d6e21e5764959b`です。
図／CSVは更新1を表示し、証拠には全8更新があります。このノートは監査済みJSON／CSVを
集約しただけで、新しいtensor／data読出しやGPU実行はしていません。

- full: `89238d2bece6f2537c68e3053e4da839a138dba7f4f78cd8c0bc92fe54faeba8`
- coordinates: `046c2e03cd9c2166379c38972cf3efb2075b35f00f903a5e40b4bc7073706fb6`
- analysis: `cd6c8311ba354f674e8abedbafc8630d57688057ecd22e584990cd0beccecc93`
- files: `3d857e153906395c76cf516f0afeb54508bf532acfa375b99e0c69f6e916ab3a`
- torus: `f95c32fc0c2b8c0875464ca1f4cd74d2fe990dc4854dde854cf4550a30502950`
