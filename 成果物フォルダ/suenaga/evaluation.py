from reinmake import reinforce_main
from parameters import make_tendency_matrix, make_similality_matrix

players = ["playerA", "playerB", "playerC", "playerD", "playerE"]

for player in range(5):
    check = reinforce_main()
    print(players[player])
    print(check[player])
    print("比較対象")
    print(make_tendency_matrix()[player])