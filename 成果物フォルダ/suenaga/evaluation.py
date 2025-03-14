# from reinmake import reinforce_main
from parameters import make_tendency_matrix, make_similality_matrix
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

def evaluation_method(check):

    players = ["playerA", "playerB", "playerC", "playerD", "playerE"]
    sites = ["SiteA", "SiteB", "SiteC", "SiteD", "SiteE"]

    save_locale = "many_learning_results"
    save_para = "a09_g09_e09"


    # 類似率の行列（人間Aは傾向Aとどれくらい似ているのか？） 行　人間、　列　傾向
    similarity_matrix = make_similality_matrix()

    # # ある特徴を持つ人間のサイト遷移確率　（傾向Aから傾向E）　5つの要素は傾向AからE　行　サイト、　列　サイト
    tendency_tables = make_tendency_matrix()

    reward_table = np.einsum('ij,jkl->ikl', similarity_matrix, tendency_tables)

    for player in range(5):
        # check = reinforce_main()
        print(players[player])
        # print(check[player])
        print(np.round(check[player], 2))
        print("比較対象")
        print(reward_table[player])

        for i in range(5):
            q_data = check[player][i]
            # データの最小値と最大値を取得
            min_value = np.min(q_data)
            max_value = np.max(q_data)
            print(q_data)
            print(max_value)
            print(min_value)

            # データを0から1の範囲に正規化
            normalized_q_data = (q_data - min_value) / (max_value - min_value)

            plt.plot(normalized_q_data, label='学習後のQテーブル')
            plt.plot(reward_table[player][i], label='類似した傾向が次のサイトへ遷移する確率')
            plt.title(f"{players[player]}が{sites[i]}にいるとき、次のサイトへ遷移する期待値")
            plt.xlabel("次に遷移するサイト")
            plt.ylabel("Q値")
            plt.xticks(ticks=range(5), labels=["サイトA", "サイトB", "サイトC", "サイトD", "サイトE"])
            # plt.yticks(ticks=range(5), labels=["", "サイトB", "サイトC", "サイトD", "サイトE"])
            plt.grid(True)
            plt.legend(loc='upper right')

            save_path = f"C:/Users/N25845/Desktop/TM59_git_folder/2024_TM-59/成果物フォルダ/suenaga/picture/{save_locale}/{save_para}/{players[player]}_{sites[i]}.png"
            # plt.savefig(save_path)
            # plt.close()
            plt.show()