# 中心活動型Polar：限定FFNパイロット

[English (canonical)](ffn-center-activity-20261107.md) | 日本語

振幅側の活動で物理半径を制御する意味を問い直し、ユーザーはA100での最初の比較を認めました。
(s,t)、振幅の角度方向の提案、L/U、二次モーメントを合わせた初期化を維持し、
活動源と明示したgainだけを変えます。新しい座標embeddingやsigmoid/clip/smoothstep目標制御の採用ではありません。

既存TinyStoriesを使い、幅256、pre-LN4ブロック・4 head、context128、FFN4N、seed17、
各FFN射影A=ceil(8 sqrt(d_in d_out))=4096、batch2・累積4（1024 token/更新）、
eager FP32・TF32無効、AdamW(.9,.95)、weight decay0、global clip1、
baseLR3e-4・Atom全体LR16倍、LRとデータ順序のhorizon20M tokenに固定します。
各条件400更新・409600 token、DEV8192 tokenのみで、confirmationや生成選択はしません。

包絡は既存の開発実験で選んだminimum基準のrational_zero_opening、birth=minimum=.5、
upper_floor1、maximum50、epsilon=.001、初期p0に固定します。過去400更新の観察から選びましたが、
事前固定したheldoutの発見や最適包絡の主張ではありません。

fresh DenseとCST4条件を実行します。CSTは振幅活動・gain1、中心活動center_chord・gain1/64/4096です。
全CSTで同じ明示登録Torch radius eager更新Algorithm v1と同じnative Linear Planを使い、
更新backendの違いを混ぜません。gainは活動の尺度の探索で、base/field LRは変えません。
新しい学習前に固定し、このbatchを自動延長したりheldoutの勝者を選んだりしません。

中心活動はE=.5[(shortestDeltaMu_out/r_out_old)^2+(shortestDeltaMu_in/r_in_old)^2]、
S*=clip(S+gain*E,1,4)、その後に従来の厳密半径減衰を使います。S=||s,t||²、p=(S-1)/3です。
追加EMA・弱Atom拡張・optimizer時間での除算は使いません。小さな移動が続いても必ずpが1になるとは限りません。
表現を保って更新policyを変える比較で、実移動量・学習率・support規格化が解釈に影響します。

学習前にGPU識別・空きメモリ、ソース・データhash、CST4条件の初期Parameter・初期化report一致、
CPU/CUDA更新テスト、独立quadrature、native長方形Y/dX/全Atom VJP、
mixed optimizer moment/resume、実際の4ブロックtask/resumeを各recipeで検査します。
失敗した凍結attemptは停止し、そのsnapshotを編集して続行しません。

train/DEV曲線、同期したoptimizer更新全体の時間、allocated/reserved peak、setup/wall time、
値・勾配probe、振幅別の半径・p・L/Uのcountと分位点を保存します。中心の最短移動は全更新で
読み取り専用に観測し、時間計測外とします。CST各条件26時点のgeometryを保持し、
既存診断のp/q名とAtom密度を区別します。

学習後は各CST条件8射影のnative Y/dX/全Atom VJP、DEV完全再計算、
復元後の次task更新Parameterとoptimizer状態の完全一致、最終geometryとcheckpoint一致を検査します。
compact raw evidenceを回収しmanifest/hashを照合し、元のソースとcheckpointを残します。
研究更新はGraph captureを拒否するためGraphの主張はしません。時間・メモリも限定eager cohortの観測で、採用結果ではありません。

ソースrevisionとデータSHAは[登録情報](../src/cst_research/studies/ffn-center-activity-20261107.json)に固定します。
runtime変更はtorchcst、consumer・qualification・controllerは研究repoに置き、
接続コマンドやraw receiptはignored evidenceに保存します。
