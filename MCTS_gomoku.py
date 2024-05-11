from abc import ABC, abstractmethod
import numpy as np
import networkx as nx

class GamePlayerInterface(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def move(self, board):
        pass

    @abstractmethod
    def new_game(self, side):
        pass

class MCTSAlgorithm(GamePlayerInterface):

    WIN_VALUE = 1
    DRAW_VALUE = 0
    LOSS_VALUE = -1

    def __init__(self, c=1.414, epoch=750, simul_num=30):
        super().__init__()
        self.side = None
        self.tree = nx.DiGraph()
        self.C = c
        self.epoch = epoch
        self.simul_num = simul_num

    def new_game(self, side):
        self.side = side
        self.tree.clear()

    def check_num(self, move, side, original_state):
        four_stones = []
        row = move//8
        col = move%8
        finish = False

        def check_direction(_dr, _dc):
            count = 0
            r, c = row + _dr, col + _dc
            p = r * 8 + c
            indices = []
            margin = 2
            while 0 <= r < 8 and 0 <= c < 8 and original_state[p] == side:
                count += 1
                r += _dr
                c += _dc
                p = r * 8 + c
            while 0 <= r < 8 and 0 <= c < 8 and original_state[p].value == 0 and margin:
                margin-=1
                indices.append(p)
                r += _dr
                c += _dc
                p = r * 8 + c
            return count, indices

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            total_count = 1
            count_1, indices_1 = check_direction(-dr, -dc)
            count_2, indices_2 = check_direction(dr, dc)
            total_count += count_1
            total_count += count_2

            if total_count >= 5:
                finish = True
                return finish, four_stones

            if total_count == 4 and indices_1 and indices_2:
                four_stones+=indices_1
                four_stones+=indices_2

        return finish, four_stones

    def move(self, board):
        root_hash = board.hash_value()
        original_state = board.state.copy()
        numeric_state = np.vectorize(lambda x: x.value)(original_state)
        empty_indices = np.flatnonzero(numeric_state == 0)
        possible_moves = len(empty_indices)
        opp_side = board.other_side(self.side)

        init_guid = []
        very_urgent = []
        urgency = []

        if possible_moves == 64:
            location = 28
            _, res, finished = board.move(location, self.side)
            return res, finished

        if 64-1 >= possible_moves >= 64-3:
            locked_indices = [index for index in range(64) if index not in empty_indices]
            for move in locked_indices:
                rr, cc = move//8, move%8
                for dx in range(-2,3):
                    nr = rr+dx
                    if 0<=nr<8:
                        for dy in range(-2,3):
                            nc = cc+dy
                            if 0<=nc<8:
                                init_guid.append(nr*8+nc)
            empty_indices = [item for item in set(init_guid) if item not in locked_indices]

        if possible_moves<=64-5:
            for move in empty_indices:
                finish, four_stones = self.check_num(move, self.side, original_state)
                if finish:
                    _, res, finished = board.move(move, self.side)
                    return res, finished
                if four_stones:
                    urgency.append(move)
                    urgency+=four_stones

                finish, four_stones = self.check_num(move, opp_side, original_state)
                if finish:
                    very_urgent.append(move)
                if four_stones:
                    urgency.append(move)
                    urgency+=four_stones

            urgency = list(set(urgency))
            if len(very_urgent)==1:
                move = very_urgent[0]
                _, res, finished = board.move(move, self.side)
                return res, finished

        if very_urgent:
            empty_indices = very_urgent
        elif urgency:
            empty_indices = urgency

        self.tree.add_node(root_hash, total_return=0, N = possible_moves * self.simul_num, index=None)

        for index in empty_indices:
            new_hash = root_hash + self.side.value * 8**(63-int(index))
            self.tree.add_node(new_hash, total_return=0, N = self.simul_num, index = index)
            self.tree.add_edge(root_hash, new_hash)
            result = 0
            for _ in range(self.simul_num):
                res = self.simulate(board, index)
                result += res
                board.state[:] = original_state
            self.tree.nodes[new_hash]['total_return'] += result

        for _ in range(self.epoch-possible_moves):
            selected_node_hash = self.select(root_hash)
            location =  self.tree.nodes[selected_node_hash]['index']
            result = 0
            for _ in range(self.simul_num):
                res = self.simulate(board, location)
                result += res
                board.state[:] = original_state
            self.tree.nodes[selected_node_hash]['total_return'] += result
            self.tree.nodes[selected_node_hash]['N'] += self.simul_num
            self.tree.nodes[root_hash]['N'] += self.simul_num

        best_move, best_value = None, float('-inf')
        for child_hash in self.tree.successors(root_hash):
            child_node = self.tree.nodes[child_hash]
            value = child_node['total_return'] / child_node['N']
            if value > best_value:
                best_move, best_value = child_node['index'], value
        _, res, finished = board.move(best_move, self.side)
        return res, finished

    def select(self,root_hash):
        ucb_values = []
        root_hash_data_N = self.tree.nodes[root_hash]['N']
        for child_hash in self.tree.successors(root_hash):
            ucb_value = self.calculate_ucb(child_hash, root_hash_data_N)
            ucb_values.append((child_hash, ucb_value))
        best_hash = max(ucb_values, key=lambda x: x[1])[0]
        return best_hash

    def calculate_ucb(self, node_hash, root_hash_data_N):
        node_data = self.tree.nodes[node_hash]
        visit = node_data['N']
        total_return = node_data['total_return']
        ucb_value = (total_return / visit) + self.C * np.sqrt(2 * np.log(root_hash_data_N) / (visit))
        return ucb_value

    def simulate(self, board, location):
        side = self.side
        _, res, finished = board.move(location, side)
        step = 0
        while not finished:
           location = board.random_empty_spot()
           side = board.other_side(side)
           _, res, finished = board.move(location, side)
           step +=1
        if res.value != 3:
            res = 1 if (self.side.value == res.value) else -1
            return res * 0.998**step
        return 0