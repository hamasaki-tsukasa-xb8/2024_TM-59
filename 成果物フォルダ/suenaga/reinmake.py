import numpy as np
import random
from parameters import make_tendency_matrix, make_similality_matrix

class RecommenderAgent:
    def __init__(self, sites, q_table=None, alpha=0.1, gamma=0.9, epsilon=0.6):
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
        # print('self.alpha=', self.alpha)
        # print('reward=',reward)
        # print('self.gamma=',self.gamma)

        # print('self.q_table=',self.q_table)

        self.q_table[state_index, action_index] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[state_index, action_index])
        # print(self.q_table[state_index, action_index])
        # print(self.q_table)

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

    # 各プレイヤーに対してエージェントのインスタンスを作成
    # agents = {player: RecommenderAgent(sites, q_table=q_tables[idx]) for idx, player in enumerate(players)}
    agents = {player: RecommenderAgent(sites) for player in players}

    # # シミュレーション（例）
    for episode in range(1000):  # 1000回の学習を行う
        for player_idx, player in enumerate(players):
            agent = agents[player]
            current_site = random.choice(sites)  # ランダムに開始サイトを選択
            
            for step in range(10):  # 各エピソードで最大10ステップ
                # 次のサイトを推薦
                next_site = agent.recommend(current_site)
                
                # print(f"Player: {player}, Current Site: {current_site}, Recommended Site: {next_site}")
                
                # ユーザーが次のサイトをクリックするかどうかをシミュレート（仮に50%の確率でクリックする）
                # random_value = random.random()
                # print(random_value) 
                # reward = 1 if random_value < 0.5 else 0
                
                # 状態遷移とQテーブルの更新
                current_site_index = agent.get_state_index(current_site)
                next_site_index = agent.get_state_index(next_site)
                action_index = agent.get_state_index(next_site)

                reward = reward_table[player_idx,current_site_index,next_site_index]
                
                # Q値を更新
                agent.update_q_table(current_site_index, action_index, reward, next_site_index)
                
                # 次の状態に遷移
                current_site = next_site



    # 各プレイヤーの学習後のQテーブルを表示
    qtable_box = []
    player_box = []
    for player in players:
        # print(f"学習後のQテーブル ({player}):")
        # print(agents[player].q_table)
        add = agents[player].q_table
        qtable_box.append(add)
    print(np.array(qtable_box).shape)
    # print(qtable_box)
    return qtable_box

reinforce_main()

