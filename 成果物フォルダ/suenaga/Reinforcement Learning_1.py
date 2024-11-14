import numpy as np
import random

class RecommenderAgent:
    def __init__(self, sites, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.sites = sites
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.epsilon = epsilon  # ε-greedy戦略
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

# サイトのリスト（例）
sites = ["SiteA", "SiteB", "SiteC", "SiteD", "SiteE"]

# エージェントのインスタンスを作成
agent = RecommenderAgent(sites)

# シミュレーション（例）
for episode in range(1000):  # 1000回の学習を行う
    current_site = random.choice(sites)  # ランダムに開始サイトを選択
    for step in range(10):  # 各エピソードで最大10ステップ
        # 次のサイトを推薦
        next_site = agent.recommend(current_site)
        print(f"Current Site: {current_site}, Recommended Site: {next_site}")
        
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

print("学習後のQテーブル:")
print(agent.q_table)
