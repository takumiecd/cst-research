# 実出力RMSを揃えた広い初期fractionの比較

[English](ffn-probe-rms-start-20261109.md) | 日本語

直前のmoment一致試行は初期化の診断として保全し、広い開始条件への否定材料にはしない。
期待Atom energyを合わせても、実FFN出力RMSがfraction.5でDenseの4倍、1で42倍となった。
構造化重みと相関・非ゼロ平均の隠れ特徴では、独立重みを仮定したtraceだけの分散近似が
不十分な可能性がある。観測と整合する機構仮説で、共分散の監査完了を意味しない。

ユーザーが希望したスケール制御を直接行う。公開moment初期化から出発し、初期化時の
振幅だけを変更する。固定TRAIN window（token0..127、ラベル不使用）で同seedのDenseから
FFN up/down8箇所の実出力RMS目標を取る。forward順に各射影の実入力を固定し、
正の振幅倍率をbracket/bisectionで探す。上限32を保ち、fraction・中心はbit一致、通常Parameterは
変更しない。最後に全8箇所の実RMSが目標の0.1%以内であることを必須とする。
到達できなければ全振幅・reportを復元して学習前に停止し、閾値を緩めない。
符号化された階層の切替により、単一倍率では届かない目標もある。
Norm層・runtime gain・永続controller・ラベルに依存するfitは追加しない。

各箇所の倍率、前後RMS、実振幅、元のmoment reportを残す。校正後の期待重みmomentは
元の目標と異なり得るため、両方を同時に一致させたとは主張しない。
別の非重複TRAIN window（token128..255）でもRMSを測るが、倍率選択には使わない。
固定校正windowに対する制御であり、全入力や置換していないAttention出力の一致ではない。
校正入力・data・sourceのhashを保全する。

共通振幅上限32、幅512、4pre-normブロック、8head、context128、expansion4、各FFN8192Atom。
attention/embedding/共有語彙headはDense。校正ありの主6条件は
(tau,fraction)=(.003,0),(.001,0),(.0003,0),(.003,.25),(.003,.5),(.003,1)。
新規Denseと、校正なしの上限32/.003/fraction0 MOMENT-only bridgeも実行する。
seed17、段階1は全8条件256更新/262144tokens。学習前に、小tau2条件の有限DEV最低条件、
広いfraction3条件の同様の最低条件を別々に選ぶ規則を固定（同率は小さいtau/fraction）。
段階2はその2条件、校正あり共通baseline、Denseを最初から976更新/999424tokens。
合計12本の限定探索であり、収束・3seed確認・普遍的最適値・半径の因果効果は示さない。

従来20M-tokenのschedule/データ順序horizon、warmup200、batch2×128×accumulation4、
baseLR3e-4、betas(.9,.95)、eps1e-8、decay0、clip1、振幅/中心16倍、fraction1倍を維持。
fractionの動きが遅い可能性は残り、初期化とoptimizer rateは別の問い。
Parameter/moments/物理行FP32、写像中間FP64、TF32無効。
検証済みtoken manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`を使い、
選択はDEV8192対象のみ。confirmationは使わない。

operator0d277ae6は同じRTX6000 Ada/Torch2.9.1/cu128で68GPU検査済みのsourceとbyte一致を
確認して維持し、旧結果hashも検証して再利用する。consumer/configは新しく検査し、
実校正、最終0.1%RMS、写像/Jacobian、別forwardの保持、有限な更新/fraction変化、
fraction.5/1の保存復元後の厳密継続を確認。source/初期化modeを新receiptに含め、
moment-only receiptで校正済みrunを承認しない。

学習・半径・fraction・振幅bin・各勾配の図に加え、実activation RMSと別window比率を残す。
更新時間とallocated/reserved peakの範囲は従来同様（zero_grad/LR設定/probe/評価を除く）。
準備・校正wall時間を別記録。Graph・process全体メモリは未主張。
全12checkpointでDEVを1e-6以内に再現し、counter、FP32 moments有限性、実上限を確認。
学習済みup/downの非連続7行Y/dX/全Atom VJPをFP32物理行/FP64参照へ既存atol1e-4/rtol4e-3で
照合する。compact証拠をhash検証して回収し、checkpointをremoteに保全、所有job終了を確認する。
