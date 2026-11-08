# 小さいtauと広い初期radius_fraction

[English](ffn-radius-fraction-start-20261109.md) | 日本語

完了した上限1のtau比較と別の探索studyとし、旧凍結試行を書き換えない。
ユーザーは小さいtauと広い初期fractionを提案し、同じDense初期moment目標を保って
全条件共通の振幅上限を広げる方針を選んだ。8192Atom・上限1ではfraction.5/1の
従来up目標分散に届かない。上限32なら実現でき、fraction1/半径50のup振幅RMSは
約10になる。広い値域には新しいnative検査が必要。

主条件はすべてamplitude_max32、明示backend levels_wide_amplitude：
(tau,初期fraction)=(.003,0),(.001,0),(.0003,0),(.003,.25),(.003,.5),(.003,1)。
上限1/.003/fraction0の新規bridgeと新規Denseも加える。seed17のみ。
幅512、4pre-normブロック、8head、context128、expansion4、各FFN射影8192Atom。
attention/embedding/共有語彙headはDense。通常重み・中心・振幅Gaussian乱数streamを共通化。
公開moment初期化でup分散.0004、down.00005、相対誤差.005以内に合わせる。
期待energyの一致であり、サンプルごとの実activation RMSの厳密一致ではない。
固定probeの出力RMS・loss・各勾配も記録し、相関の影響を見えるようにする。

段階1は全8条件256更新/262144tokens。結果を見る前に選択規則を固定する。
小tau2条件から最終DEVが有限かつ最低のもの、広いfraction3条件から同様に最低のものを
別々に選ぶ（同率は小さいtau/fraction）。段階2はその2条件と共通上限32/.003/fraction0、
Denseを最初から976更新/999424tokens。合計12本の限定探索で、3seed確認ではない。
正しさの検査に失敗した場合は学習前に停止し記録を残す。学習が失敗・非有限の条件は選外とし、
そのfamilyを拡張する前に原因を調べる。追加の自動sweepや公開デフォルト変更は行わない。

従来20M-tokenのデータ順序/LR horizon、warmup200、batch2/context128/accumulation4、
AdamW baseLR3e-4、betas(.9,.95)、eps1e-8、decay0、globalclip1を保つ。
振幅/中心16倍、fraction1倍を固定。広いfractionは動きが遅い可能性があり、短期の不調から
広い初期値自体が悪いとは言えない。Parameter/moments/物理行FP32、写像中間FP64、TF32無効。
検証済みtoken manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`のみ再利用。
選択はDEV8192対象、confirmationは使わない。旧曲線は文脈に留め、新規cap32/cap1対照を
実行し、過去の速度基準を混ぜない。

学習前にCPU初期化、実GPUの|a|<=32、両方向、7/256行、半径上限4.7/50を検査。
FP32物理行を保つFP64参照のY/dX/全Atom VJPは既存atol1e-4/rtol4e-3を維持し、
retained backwardと保存復元後の次更新も検査する。全task recipeを新規検査し、
広いfraction.5/1でも保存復元を確認。新しいsourceに紐付くreceiptを使う。

更新/token/wall時間に対するtrain/DEV曲線、allocated/reserved peak、従来範囲の更新時間
（zero_grad/LR設定/probe/評価を除く）を記録。fraction・半径・32まで拡張した振幅bin・
端点割合・フィールド勾配・実activation RMSを描く。初期振幅の上限飽和も比較する。
RMSが同じでも特徴の多様性や隠れた相関は一致しない。振幅binは現在値で分類し、固定集団の
軌跡ではない。収束・最適半径の因果性・スケールのオーダーは主張しない。

学習後は長い4checkpointのDEV CEを1e-6以内で再現し、optimizer counter、FP32 momentsの
有限性、実amplitude_maxに対する全Atom範囲を確認。CST長実行ごとに学習済みblock0 up/downの
非連続7行Y/dX/全Atom勾配をFP32物理行/FP64参照へ同じ閾値で照合する。
source/data/checkpoint/result hashを確認し、compact証拠を回収、大きいcheckpointをremoteに保持。
所有processの終了を確認し、source/operator/job IDと最初の失敗を別々に残す。
