from reinmake import reinforce_main
from parameters import make_tendency_matrix, make_similality_matrix
import numpy as np

players = ["playerA", "playerB", "playerC", "playerD", "playerE"]


# 類似率の行列（人間Aは傾向Aとどれくらい似ているのか？） 行　人間、　列　傾向
similarity_matrix = make_similality_matrix()

# ある特徴を持つ人間のサイト遷移確率　（傾向Aから傾向E）　5つの要素は傾向AからE　行　サイト、　列　サイト
tendency_tables = make_tendency_matrix()

reward_table = np.einsum('ij,jkl->ikl', similarity_matrix, tendency_tables)

for player in range(5):
    check = reinforce_main()
    print(players[player])
    print(check[player])
    print("比較対象")
    print(reward_table[player])