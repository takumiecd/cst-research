# 半径の包絡線を実装する前のworktree整理

[English (canonical)](worktree-cleanup-20261108.md) | 日本語

別デバイスでの独立した復元検証後、停止中のworktreeを11個削除しました。
torchcstが6個、cst-researchが5個です。両repoともmainと現在の開発用checkoutを1つずつ残しました。
未マージのブランチとPRは保持しています。mainへのpush、merge、remote branchの削除、
共通の実験outputの削除は行っていません。

削除前に、対象すべてのtracked/untracked状態がcleanであることを確認しました。
到達可能な全Git ref・履歴をrepo別bundleへ保存し、各rootの.gitポインタを除く
worktree全体を、ignoredファイル・symlinkの参照先も含めて保存しています。
11613エントリ、ファイルの論理サイズ284691846 bytesで、worktree archiveは107362633 bytesです。
別途bundleがtorchcst18116055 bytes、cst-research18417901 bytesあります。

WindowsのEドライブ側で独立に展開し、全ファイルのSHA256とlink先を照合しました。
両bundleのmirror clone、branch HEADの一致、git fsckも成功しました。
削除直前にローカルのHEAD・statusと全保存ファイルを再検査しました。
削除対象worktreeを作業ディレクトリに使うプロセスもありませんでした。
[コンパクトな検証記録](worktree-cleanup-20261108.json)を残しています。

## 復元

Archive IDは`20261108-cst-worktrees-01`です。所有者のローカル復元ディレクトリは
`torchcst/output/worktree-cleanup-20261108-01/`で、README.md、manifest.json、verified.json、
retirement.json、source bundle、worktrees.tar.gzがあります。
そのREADMEに別デバイス側の正確な保存先を記録しています。Windows側にも独立検証済みの
展開済みディレクトリとmirror repositoryを保持しています。
これらはprivateな復元記録であり、マシン固有の絶対パスやraw payloadはこのrepoへcommitしません。

**新しい場所**へ対象bundleをmirror cloneし、manifestから必要なbranch/HEADを確認して
worktreeを追加します。対応するarchive内ディレクトリだけを重ねるとignoredファイルも復元できます。
symlink先は記録どおり保持しているため、別ホストでは再作成が必要な場合があります。
現在の開発checkoutへ上書きしないでください。

torchcstの開発checkoutは`lm-attention-levels`から`radius-envelope`へ移し、
`feat/radius-envelope`へ切り替えました。従来の`kernel/lm-attention-levels`のrefと
Draft PR155は保持しています。研究checkoutは`lm-attention-output`、
`feat/lm-attention-output`/Draft PR35を継続します。
過去のsource commitとremoteの凍結済み実験ディレクトリは変更していません。
記録された古いローカルcheckoutパスも、対応refから再作成できます。

## 選択した次の設計

API名は所有者との議論を踏まえて`kernels.radius_envelope`としました。
質量としての振幅aを保ち、各Atomのqを独立に学習します。Tauで下限の形だけを調整し、
U=r_max、L=r_min+(r_max-r_min)exp(-(a/tau)^2)、r=L^(1-q)U^qとします。
この実装には、中心移動量による制御やエネルギー振幅への置き換えは入れません。
数学的宣言、参照実行、勾配、optimizerの射影はtorchcstが管理し、将来のLM研究は
このrepoが管理します。実装検証は新しいLM学習結果や最適tauの主張ではありません。

実装は[torchcst PR156](https://github.com/takumiecd/torchcst/pull/156)に保全しています。
PR155を親にした未マージのPRです。最終source40e120c9f1b7260364f08bf6723185f62bf31ac6で、
RTX6000 Ada上の40検査（実CUDA14・CPU/metadata26）が成功しました。最大半径4.7/50、
両FFN方向・7/256行、qの勾配と更新、保存したモデルとAdamの次の更新の一致を含みます。
同じruntimeのa86db78d時点のローカルCPU一式は2433成功・2408skip、最終対象CPUは24成功・CUDA8skipです。
環境エラーと、精度形式が一致しなかった参照の修正も保全し、ee9c8955以降のruntime sourceは不変です。
演算の検証記録はtorchcstで管理し、将来のLM学習研究と分けています。
この更新で新しいregistry studyや学習結果は宣言しません。
