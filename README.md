論文に書かれていたBitNet1.58bのBitLinear部の実装です。<br>
https://arxiv.org/abs/2310.11453<br>
https://arxiv.org/abs/2402.17764<br>
<br>
論文を読み自力で書いたところもありますが、以下の方々のコードが非常に参考になりました。<br>
https://github.com/Beomi/BitNet-Transformers<br>
https://github.com/frodo821/BitNet-Transformers<br>
https://github.com/kyegomez/BitNet<br>
<br>
参考にさせてもらったコードから大きな変更点としては、activationの順番とdeqauntizeの部分を直しています。<br>
また、Beomiさんのコードのllamaに追加してみたところ、lossは4.1付近でした。(epoch50000ほど)<br>
何か問題があった場合は教えてください。<br>

追記:実験結果と同様の結果が得るためにモデルサイズを上げたのですが、自分のllamaのパラメーターの設定が悪いのか基本的にlossは上がる一方でした。

