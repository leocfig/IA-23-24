# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

import sys
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de PipeMania."""

    def __init__(self, grid):
        """Cria um objeto Board que contém uma grelha n x n."""
        self.grid = grid

    def is_grid_index(self, row: int, col: int) -> bool:
        """Devolve True se a posição do tabuleiro é válida, False caso contrário."""
        return 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0])

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if self.is_grid_index(row, col):
            return self.grid[row][col]
        else:
            return None

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        if self.is_grid_index(row, col):
            return self.get_value(row - 1, col), self.get_value(row + 1, col)
        else:
            return None # ou temos de pôr (None, None) ? caso a peça não exista

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if self.is_grid_index(row, col):
            return self.get_value(row, col - 1), self.get_value(row, col + 1)
        else:
            return None # ou temos de pôr (None, None) ? caso a peça não exista

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board."""

        grid = []
        first_line = sys.stdin.readline().split() # Lê a primeira linha para obter o tamanho da grelha
        n = len(first_line)
        grid.append(first_line)  # Adiciona a primeira linha à grelha
        for _ in range(n-1):     # Assegura que o input é n x n
            row = sys.stdin.readline().split()
            grid.append(row)
        return Board(grid)
    
    def print_grid(self):
        for row in self.grid:
            print(' '.join(row))

    # TODO: outros metodos da classe


class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:

    board = Board.parse_instance()
    board.print_grid()

    print(board.adjacent_vertical_values(0, 0))
    print(board.adjacent_horizontal_values(0, 0))

    print(board.adjacent_vertical_values(1, 1))
    print(board.adjacent_horizontal_values(1, 1))

    print(board.adjacent_horizontal_values(2, 3))
    print(board.adjacent_horizontal_values(2, 2))

    

    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
