import numpy as np
import random

class RecommenderAgent:
    def __init__(self, sites, q_table=None, alpha=0.1, gamma=0.9, epsilon=0.1):
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
        self.q_table[state_index, action_index] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[state_index, action_index])

    def recommend(self, current_site):
        # 現在のサイトに基づいて次に推薦するサイトを選択
        state_index = self.get_state_index(current_site)
        action_index = self.choose_action(state_index)
        return self.sites[action_index]

    def update_with_similarity(self, other_q_table, similarity_rate):
        # self.q_table = (1 - similarity_rate) * self.q_table + similarity_rate * other_q_table
        self.q_table = 0.5 * self.q_table + 0.5 * other_q_table

# サイトのリスト（例）
sites = ["SiteA", "SiteB", "SiteC", "SiteD", "SiteE"]

# プレイヤーのリスト
players = ["playerA", "playerB", "playerC", "playerD", "playerE"]

# 類似率の行列（例）
similarity_matrix = np.array([
    [0.0, 0.2, 0.4, 0.6, 0.8],
    [0.8, 0.0, 0.2, 0.4, 0.6],
    [0.6, 0.8, 0.0, 0.2, 0.4],
    [0.4, 0.6, 0.8, 0.0, 0.2],
    [0.2, 0.4, 0.6, 0.8, 0.0]
])

# Qテーブルの初期化
q_tables = [
    np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8],
        [0.8, 0.0, 0.2, 0.4, 0.6],
        [0.6, 0.8, 0.0, 0.2, 0.4],
        [0.4, 0.6, 0.8, 0.0, 0.2],
        [0.2, 0.4, 0.6, 0.8, 0.0]
    ]),
    np.array([
        [0.0, 0.4, 0.8, 0.2, 0.6],
        [0.6, 0.0, 0.4, 0.8, 0.2],
        [0.2, 0.6, 0.0, 0.4, 0.8],
        [0.8, 0.2, 0.6, 0.0, 0.4],
        [0.4, 0.8, 0.2, 0.6, 0.0]
    ]),
    np.array([
        [0.0, 0.6, 0.2, 0.8, 0.4],
        [0.4, 0.0, 0.6, 0.2, 0.8],
        [0.8, 0.4, 0.0, 0.6, 0.2],
        [0.2, 0.8, 0.4, 0.0, 0.6],
        [0.6, 0.2, 0.8, 0.4, 0.0]
    ]),
    np.array([
        [0.0, 0.8, 0.6, 0.4, 0.2],
        [0.2, 0.0, 0.8, 0.6, 0.4],
        [0.4, 0.2, 0.0, 0.8, 0.6],
        [0.6, 0.4, 0.2, 0.0, 0.8],
        [0.8, 0.6, 0.4, 0.2, 0.0]
    ]),
    np.array([
        [0.0, 0.2, 0.6, 0.4, 0.8],
        [0.8, 0.0, 0.2, 0.6, 0.4],
        [0.4, 0.8, 0.0, 0.2, 0.6],
        [0.6, 0.4, 0.8, 0.0, 0.2],
        [0.2, 0.6, 0.4, 0.8, 0.0]
    ])
]

# 各プレイヤーに対してエージェントのインスタンスを作成
# agents = {player: RecommenderAgent(sites, q_table=q_tables[idx]) for idx, player in enumerate(players)}
agents = {player: RecommenderAgent(sites) for player in players}

# シミュレーション（例）
for episode in range(1000):  # 1000回の学習を行う
    for player_idx, player in enumerate(players):
        agent = agents[player]
        current_site = random.choice(sites)  # ランダムに開始サイトを選択

        for step in range(10):  # 各エピソードで最大10ステップ
            # 次のサイトを推薦
            next_site = agent.recommend(current_site)
            # print(f"Player: {player}, Current Site: {current_site}, Recommended Site: {next_site}")
            
            # ユーザーが次のサイトをクリックするかどうかをシミュレート（仮に50%の確率でクリックする）
            reward = 1 if random.random() < 0.8 else 0
            
            # 状態遷移とQテーブルの更新
            current_site_index = agent.get_state_index(current_site)
            next_site_index = agent.get_state_index(next_site)
            action_index = agent.get_state_index(next_site)
            
            # Q値を更新
            agent.update_q_table(current_site_index, action_index, reward, next_site_index)
            
            # 次の状態に遷移
            current_site = next_site
        if episode < 100:
            print("================================================================================")
            print(f"Player: {player}, Current Site: {current_site}, Recommended Site: {next_site}, number: {episode}")
            print(agents[player].q_table)

        # 10回に1回、類似率が高いQテーブルと自身のQテーブルを平均する
        if episode % 10 == 0 and episode != 0:
            most_similar_idx = np.argmax(similarity_matrix[player_idx])
            # most_similar_agent = agents[players[most_similar_idx]]
            most_similar_agent = q_tables[most_similar_idx]
            similarity_rate = similarity_matrix[player_idx, most_similar_idx]
            # agent.update_with_similarity(most_similar_agent.q_table, similarity_rate)
            agent.update_with_similarity(most_similar_agent, similarity_rate)

# 各プレイヤーの学習後のQテーブルを表示
# for player in players:
#     print(f"学習後のQテーブル ({player}):")
#     print(agents[player].q_table)

