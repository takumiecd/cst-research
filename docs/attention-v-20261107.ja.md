# CSTのV射影への適用可能性を調べる短期比較

[English (canonical)](attention-v-20261107.md) | 日本語

確保済みRTX4090で、新しい全Dense対照と、4つのAttentionブロックのV射影だけを
CSTにした条件を比較します。Q/K/出力射影とFFNはDense、因果的scaled-dot-product/
softmax演算もそのままです。分離したQ/K重み・biasは名前ごとの乱数系列で作る
従来のpacked Dense初期値から厳密にコピーし、その他の共有Parameterも一致させます。
Vは公開CSTLinearと従来の有理式Polar、初期q0、c8を使用し、256×256の各射影で
2048Atomです。同じDense二次モーメント（std.02、誤差.5%以内）へ再校正します。
このデバイスでは新しい半径境界や移動量依存更新を加えません。

TinyStories token manifest SHA256は
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`。
幅256、pre-LN4ブロック、4head、context128、FFN4N、seed17、batch2/累積4、
eager FP32/TF32無効、AdamW(.9,.95)、decay0、clip1、baseLR3e-4、Atom全体LR16倍、
LR/データ順序20Mトークン予定を保ち、各条件976更新/999424トークンで終了します。
DEVは固定8192トークンです。confirmation/生成による選択、Attention全体の置換、
長期延長や自動のLR探索は行いません。

学習前に共有初期値、独立スカラー求積、実際の正方形V射影のnative Y/dX/全Atom VJP、
明示的masked-softmaxとのAttention Y/dX/Atom VJP、未来token勾配の厳密なゼロ、
optimizer moment、4ブロック実タスク/再開を検証します。学習後もV各射影の勾配、
DEV厳密再計算、最終geometryと次更新Parameter/optimizer stateの再開一致を監査します。
Dense対照も同じGPU/runtimeで新しく実行し、過去曲線は参考だけにします。

train/DEV曲線を重ね、振幅binごとの半径/q/個数を同期した更新時間の外で記録します。
更新全体の時間、準備時間、allocated/reservedピークも記録し、計測付き総経過時間には
診断の時間も含みます。ソース・データ・設定を固定し、compact rawの取得・ハッシュ検証を
行い、大きいcheckpointはリモートに保持します。1seed短期観測で収束・最適容量・
普遍的スケールや速度優位は主張しません。
