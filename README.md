論文に書かれていたbitnet158bの実装です。<br>
https://arxiv.org/abs/2310.11453<br>
https://arxiv.org/abs/2402.17764<br>
<br>
そしてこのコードは以下の方々を参考にしています。
<br>
https://github.com/Beomi/BitNet-Transformers<br>
https://github.com/frodo821/BitNet-Transformers<br>
https://github.com/kyegomez/BitNet<br>
<br>
変更点としてはdeqauntize部分を追加しています。<br>
また、Beomiさんのコードのllamaに追加してみたところ、lossは4.1付近でした。(epoch50000ほど)<br>
何か問題があった場合は教えてください。<br>
