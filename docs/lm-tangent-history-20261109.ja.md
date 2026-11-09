# 局所接空間とAdam履歴の切り分け

[English](lm-tangent-history-20261109.md) | 日本語

ユーザーが許可した**分析だけ**を実施。継続学習・LR/kernel/Atom数変更・DEV選択・改善実験・
既定値採用は行いません。幅512 LMの監査済みstep4882 checkpointと本来のタスク勾配を使用。
Dense・DST全体・CST全体のblock1 up/down、CST単独のblock1 upの計7局所写像を対象に、
実際の次のTRAIN1024対象とoffset32/64を同じ復元clockで測定（21record）。追加seed・held-outなし。

実際の入力Xを固定し、z=f(theta,X)、タスクcotangent g、J=dz/dthetaとして：

* 元のAdam・clip・field LRと通常のprojection biasを含め、微小変化Jvと実際の有限な出力変化を比較。
* 現在の勾配を反映した二次momentを固定して、一次momentだけを現在のclip後勾配へ置換。
  別に分母をabs(現在勾配)+epsilonへ置換。2x2の履歴有無、現在勾配方向、実際の丸め後変位を記録。
  方向の測定だけで、学習stepとして適用しません。境界の接方向への射影も、非制約方向と別記。
* offset0では**同じX,g**上で[X,1]のSVDから自由な線形写像の接空間を測定。Dense教師の重みを
  使うものではありません。特異値の相対cutoff1e-7。DSTは有効mask上の出力別SVDで測定。
* CSTは公開factorのCPU JVPをatom chunkごとに4回計算。Atomの独立性により各列の微分を得て、
  matrix-freeなJvとJ-transpose-gを構築。列ノルムを揃えたCGLSで-gの数値的に分解できる局所
  Atom/bias接空間への射影を推定。最大256反復、正規方程式残差目標1e-4、列cutoff1e-10。
  収束とtraceを記録し残差を明示再計算。未収束値は到達方向の下側の推定で、**構造上の上限では
  ありません**。非制約接空間はfraction/振幅の片側境界を無視し、有限stepの能力やAtom数下限でも
  ありません。range(J)にタスク方向が含まれていても、条件の悪さで現在勾配の一致度が低くなります。
* 元変位の実行可能な線分内でepsilon1/.1/.01の仮変位を評価し、周期座標をwrap。接方向と有限変化の
  角度、線形近似誤差、宣言されたrouting signatureの変化を測定。局所接空間と、有限非線形・
  routing境界・丸めの効果を分けます。モデルの学習軌跡は残しません。

native出力とfactor参照の相対L2誤差<=.005、cacheのタスクVJPとnativeのclip後勾配<=.005、
epsilon1の参照/native出力変化誤差<=.01を要求。分析用の整合基準であり、nativeのY/dX/dP基準
1e-4/4e-3は緩和しません。CPU既知解テストは明示Jacobian、条件の悪い勾配と接空間射影、
mask/自由空間、履歴なしAdamの極限、公開radius factor微分を検証します。

予約済みRTX4090でGPU/runtime/source/data/order確認とnative68件を再実施。仮更新ごとに
Parameter・buffer・optimizer・CPU/CUDA RNGを復元し、optimizer読込をdeep copy、元clockと
checkpoint hashを確認。公開operatorは固定し、研究コードは公開factorを利用してCUDA/runtimeの
数式を変えません。永続NFSへ保存し、失敗・未収束も含めcompactのhashを照合して回収。
計測処理を含む時間は性能benchmarkには使いません。

証拠:CST_RESEARCH_EVIDENCE_ROOT/lm-tangent-history-4090-20261109-01。

## 記録した分析の追加

初回21観測は通過しましたが、256反復のAtom射影は未収束で、履歴を外すとDenseでも同一batchの一致度が上がりました。別の不変snapshotで、分析用CGLSの上限を2048へ増やし、有限epsilon.001/.0001、routingが変わったAtomと変わらないAtomの線形近似誤差への寄与、各source batchの方向を全3target TRAIN batchで評価する3x3比較を追加します。同一batchへの適合と、履歴が共通方向を捉える働きを区別します。同じcheckpoint/data/operator/数値基準で、学習・改善介入は行わず、初回証拠も変更しません。

## 記録済みのフィールド・Atom別分析

所有者のフィールド・Atomに関する質問を受け、第2の不変な追加分析ではCSTの3行列状態と同じ3バッチのみを使います（9観測）。観測済みの各rawフィールドの変位とAtom全体の変位を個別に適用し、biasは固定します。有限変化・接線とタスク方向の一致度、経路切替、切替Atomと非切替Atomの線形化誤差を測ります。タスク内積をAtomごとに加算分解し、負の個数割合と絶対スコア割合を記録します。Atom別の損失変化とは解釈しません。追加の更新を適用せずAdam履歴の各条件と符号を比較します。raw振幅は包絡を通して半径も変えるため、物理的な振幅だけの分析ではありません。学習・optimizer・kernelは変更しません。
