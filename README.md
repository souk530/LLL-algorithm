# latice
承知しました。LLLアルゴリズムの数学的な説明を以下に記述します：

1. **初期化**: まず、ベクトルの基底 ${\bf b}_1, {\bf b}_2, \ldots, {\bf b}_n$ が入力として与えられます。

2. **グラム・シュミットの直交化**: このプロセスは、元のベクトルの基底から直交基底を生成します。具体的には、以下のような逐次的な手順で ${\bf b}_1, {\bf b}_2, \ldots, {\bf b}_n$ から直交基底 ${\bf b'}_1, {\bf b'}_2, \ldots, {\bf b'}_n$ を計算します:

   $${\bf b'}_i = {\bf b}_i - \sum_{j=1}^{i-1} \mu_{ij} {\bf b'}_j$$

   ここで、$\mu_{ij} = \frac{{\bf b}_i \cdot {\bf b'}_j}{{\bf b'}_j \cdot {\bf b'}_j}$ は、ベクトル ${\bf b}_i$ のベクトル ${\bf b'}_j$ への射影の大きさを表しています。

3. **サイズの削減**: このステップでは、基底ベクトルを互いに「近く」なるように調整します。具体的には、各$i$について、次の操作を繰り返します：

   $$ {\bf b}_i = {\bf b}_i - \lfloor \mu_{ij} \rceil {\bf b}_j, \quad \text{for all } j < i $$

   ここで、$\lfloor x \rceil$ は $x$ を最も近い整数に丸める操作を表しています。

4. **ベクトルペアの交換**: このステップでは、特定の条件が満たされる場合にベクトルペアを交換します。具体的には、次の条件が満たされた場合：

   $$||{\bf b'}_i||^2 > (3/4 - (\mu_{i, i-1})^2) ||{\bf b'}_{i-1}||^2$$

   ベクトル ${\bf b}_i$ と ${\bf b}_{i-1}$ を交換します。この操作後、もう一度グラム・シュミットの直交化とサイズ削減を行います。

5. **ステップ3と4の繰り返し**: ベクトルペアの交換が必要なくなる（つまり、全てのベクトルペアが上記の条件を満たさない）まで、ステップ3とステップ4を繰り返します。

以上が、LLLアル