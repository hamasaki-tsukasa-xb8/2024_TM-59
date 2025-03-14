import numpy as np
import random
from parameters import make_tendency_matrix, make_similality_matrix
from evaluation import evaluation_method
import matplotlib.pyplot as plt
import os

class RecommenderAgent:
    def __init__(self, sites, q_table=None, alpha=0.1, gamma=0.1, epsilon=0.1):
        self.sites = sites
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.epsilon = epsilon  # ε-greedy戦略
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = np.zeros((len(sites), len(sites)))  # Qテーブル

    def get_state_index(self, site):
        return self.sites.index(site)

    def choose_action(self, state_index):
        # ε-greedy戦略で次の行動を選択
        if random.uniform(0, 1) < self.epsilon:
            # ランダムに行動選択
            return random.choice(range(len(self.sites)))
        else:
            # Qテーブルに基づいて最適な行動選択
            return np.argmax(self.q_table[state_index])

    def update_q_table(self, state_index, action_index, reward, next_state_index):
        # Q値の更新
        best_next_q = np.max(self.q_table[next_state_index])
        # print('前q_table：',self.q_table)
        self.q_table[state_index, action_index] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[state_index, action_index])
        np.round(self.q_table, 5)
        # print('後q_table：',self.q_table)

    def recommend(self, current_site):
        # 現在のサイトに基づいて次に推薦するサイトを選択
        state_index = self.get_state_index(current_site)
        action_index = self.choose_action(state_index)
        return self.sites[action_index]

    def update_with_similarity(self, other_q_table, similarity_rate):
        # self.q_table = (1 - similarity_rate) * self.q_table + similarity_rate * other_q_table
        self.q_table = 0.5 * self.q_table + 0.5 * other_q_table

# サイトのリスト（例）

def reinforce_main():
    sites = ["SiteA", "SiteB", "SiteC", "SiteD", "SiteE"]

    # プレイヤーのリスト
    players = ["playerA", "playerB", "playerC", "playerD", "playerE"]

    # 類似率の行列（人間Aは傾向Aとどれくらい似ているのか？） 行　人間、　列　傾向
    similarity_matrix = make_similality_matrix()

    # ある特徴を持つ人間のサイト遷移確率　（傾向Aから傾向E）　5つの要素は傾向AからE　行　サイト、　列　サイト
    tendency_tables = make_tendency_matrix()

    reward_table = np.einsum('ij,jkl->ikl', similarity_matrix, tendency_tables)
    # print(reward_table)

    # 各プレイヤーに対してエージェントのインスタンスを作成
    # agents = {player: RecommenderAgent(sites, q_table=q_tables[idx]) for idx, player in enumerate(players)}
   
    alp=0.1
    gam=0.9
    eps=0.9
    agents = {player: RecommenderAgent(sites, q_table=None, alpha=alp, gamma=gam, epsilon=eps) for player in players}

    learning_num = 1000


    if learning_num > 101:
        save_locale = "many_learning_results"
    else:
        save_locale = "few_learning_results"
    save_para = f"a{str(alp)[0]}{str(alp)[2]}_g{str(gam)[0]}{str(gam)[2]}_e{str(eps)[0]}{str(eps)[2]}"
    # print(save_para)

    # # シミュレーション（例）
    for player_idx, player in enumerate(players):
        for episode in range(learning_num):  # 1000回の学習を行う
            agent = agents[player]
            current_site = random.choice(sites)  # ランダムに開始サイトを選択
            
            for step in range(10):  # 各エピソードで最大10ステップ
                # 次のサイトを推薦
                next_site = agent.recommend(current_site)
                
                # 状態遷移とQテーブルの更新
                current_site_index = agent.get_state_index(current_site)
                next_site_index = agent.get_state_index(next_site)
                action_index = agent.get_state_index(next_site)

                reward = reward_table[player_idx,current_site_index,next_site_index]

                # if reward > 0.89:
                #     reward += 100.0
                # Q値を更新

                agent.update_q_table(current_site_index, action_index, reward, next_site_index)
                
                # 次の状態に遷移
                current_site = next_site

        # print(player)
        # # print(agent.q_table)
        # print(np.round(agent.q_table, 5))
        # print("比較対象")
        # print(reward_table[player_idx])
        
        # q_data = agent.q_table -np.min(agent.q_table) 
        # print(q_data)
        # # データの最小値と最大値を取得
        # min_value = np.min(q_data)
        # max_value = np.max(q_data)
        # print(q_data)
        # print(max_value)
        # print(min_value)

        # # データを0から1の範囲に正規化
        # normalized_q_data = (q_data - min_value) / (max_value - min_value)
        # normalized_q_data = q_data
        # normalized_q_data -= min_value
        # print(reward_table.shape)
        # print(reward_table)

        for site_index in range(5):
            q_data = agent.q_table[site_index] -np.min(agent.q_table[site_index]) 
            # print(q_data)
            # データの最小値と最大値を取得
            min_value = np.min(q_data)
            max_value = np.max(q_data)
            # print(q_data)
            # print(max_value)
            # print(min_value)

            # データを0から1の範囲に正規化
            normalized_q_data = (q_data - min_value) / (max_value - min_value)
            # print(normalized_q_data)
            plt.plot(normalized_q_data, label='学習後のQテーブルを正規化したもの')
            plt.plot(reward_table[player_idx, site_index, :], label='類似した傾向が次のサイトへ遷移する確率')
            plt.title(f"{player} in {sites[site_index]}")
            plt.xlabel("次に遷移するサイト")
            plt.ylabel("Q値(or確率)")
            plt.xticks(ticks=range(5), labels=["サイトA", "サイトB", "サイトC", "サイトD", "サイトE"])
            # plt.yticks(ticks=range(5), labels=["", "サイトB", "サイトC", "サイトD", "サイトE"])
            plt.grid(True)
            plt.legend(loc='upper right')
        

            # save_path = f"C:/Users/N25845/Desktop/TM59_git_folder/2024_TM-59/成果物フォルダ/suenaga/picture/{save_locale}/{save_para}/{player}_{sites[site_index]}.png"
            save_midpath = f"C:/Users/N25845/Desktop/TM59_git_folder/2024_TM-59/成果物フォルダ/suenaga/other_case_picture/{save_para}"
            save_path = f"C:/Users/N25845/Desktop/TM59_git_folder/2024_TM-59/成果物フォルダ/suenaga/other_case_picture/{save_para}/{player}_{sites[site_index]}.png"
            if not os.path.exists(save_midpath):
                os.makedirs(save_midpath)
            plt.savefig(save_path)
            plt.close()
            # plt.show()


    # print("Tendency Transition Probabilities (5x5x5 matrix):")
    # print(tendency_tables)    


    print("fin") 

    # 各プレイヤーの学習後のQテーブルを表示
    # qtable_box = []
    # # player_box = []
    # for player in players:
    #     # print(f"学習後のQテーブル ({player}):")
    #     # print(agents[player].q_table)
    #     add = agents[player].q_table
    #     qtable_box.append(add)
    # # print(np.array(qtable_box).shape)
    # # print(qtable_box)
    # return qtable_box

reinforce_main()

# evaluation_method(reinforce_main())
# a = np.array(reinforce_main())
# print(a.shape)
# print(type(reinforce_main()))


