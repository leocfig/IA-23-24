# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 57:
# 106157 Leonor Costa Figueira
# 106322 Raquel dos Anjos Santos Caldeira Rodrigues

import sys, copy
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

from typing import List #ig?


class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def get_board(self):
        """Retorna o tabuleiro."""
        return self.board

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        """Método para verificar se dois estados são iguais."""
        return isinstance(other, PipeManiaState) and self.board == other.board

    def check_neighbors(self, row, col):
        """Tenta tornar a peça na posição especificada permanente com base nas vizinhas já permanentes."""

        permanent_piece = self.get_board().is_permanent(row, col)
        if permanent_piece:
            return

        # Define the indices of the neighboring positions
        neighbor_indices = [
            (row - 1, col),  # Above
            (row + 1, col),  # Below
            (row, col - 1),  # Left
            (row, col + 1)   # Right
        ]

        connections = {
            'LV': ['up', 'down'],
            'VB': ['left', 'right'],
            # Add more connections as needed for other orientations
        }

        for neighbor_row, neighbor_col in neighbor_indices:

            neighbor_piece = self.get_value(neighbor_row, neighbor_col)
            neighbor_permanent = self.is_permanent(neighbor_row, neighbor_col)
            
            if neighbor_permanent:
                #TODO
                pass


    def change_neighbors(self):
        """Faz uma interpretação do tabuleiro e faz alterações
        em vizinhos de peças permanentes no tabuleiro."""
        
        board = self.get_board()
        board_dim = len(board.grid)
        nr_iterations = board_dim // 2 if board_dim % 2 == 0 else board_dim // 2 + 1

        for i in range(1, nr_iterations): # Para ignorar a fronteira exterior, essa já foi tratada
            # Apply to the first row of the current border
            for col in range(board_dim):
                self.check_neighbors(i, col)

            # Apply to the last row of the current border               #REPETIDO
            for col in range(board_dim):
                self.check_neighbors(board_dim-1-i, col)

            # Apply to the first column of the current border
            for row in range(board_dim):
                self.check_neighbors(row, i)

            # Apply to the last column of the current border
            for row in range(1, board_dim - 1):
                self.check_neighbors(row, board_dim-1-i)



    def generate_actions(self):
        """Faz uma interpretação do estado atual do tabuleiro e gera ações possíveis."""
        possible_actions = []
        board = self.get_board()
        board_dim = len(board.grid)

        rotation_mappings_first_row_first_col = {'C': ['B', 'D'], 'E': ['B', 'D'], 'B': ['D'], 'D': ['B']}

        rotation_mappings_first_row_not_first_last_col = {
            'F': {'C': ['B', 'E', 'D'], 'E': ['B', 'D'], 'B': ['E', 'D'], 'D': ['B', 'E']},
            'V': {'C': ['B', 'E'], 'E': ['B'], 'B': ['E'], 'D': ['B', 'E']}
        }

        rotation_mappings_first_row_last_col = {'C': ['B', 'E'], 'E': ['B'], 'B': ['E'], 'D': ['B', 'E']}

        rotation_mappings_not_first_last_row_first_col = {
            'F': {'C': ['B', 'D'], 'E': ['B', 'C', 'D'], 'B': ['C', 'D'], 'D': ['B', 'C']},
            'V': {'C': ['B', 'D'], 'E': ['B', 'D'], 'B': ['E'], 'D': ['B']}
        }

        rotation_mappings_last_row_first_col = {'C': ['B'], 'E': ['C', 'D'], 'B': ['C', 'D'], 'D': ['C']}

        rotation_mappings_not_first_last_row_last_col = {
            'F': {'C': ['B', 'E'], 'E': ['B', 'C'], 'B': ['C', 'E'], 'D': ['B', 'C', 'E']},
            'V': {'C': ['E'], 'E': ['C'], 'B': ['C', 'E'], 'D': ['C', 'E']}
        }

        rotation_mappings_last_row_not_first_last_col = {
            'F': {'C': ['E', 'D'], 'E': ['C', 'D'], 'B': ['C', 'D', 'E'], 'D': ['C', 'E']},
            'V': {'C': ['D'], 'E': ['C', 'D'], 'B': ['C', 'D'], 'D': ['C']}
        }

        rotation_mappings_last_row_last_col = {'C': ['E'], 'E': ['C'], 'B': ['C', 'E'], 'D': ['C', 'E']}

        
        for row in range(len(self.board.grid)):
            for col in range(len(self.board.grid[0])):
                piece = self.board.get_value(row, col)
                print("Coordenadas: (", row, ",", col, ") Peça atual: ", piece)

                if board.is_permanent(row, col):
                    pass
                elif row == 0 and col == 0 and piece[0] == "F":
                    print("elif 1")
                    possible_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_first_row_first_col[piece[1]]])
                elif row == 0 and col != 0 and col != board_dim - 1:
                    print("elif 2")
                    possible_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_first_row_not_first_last_col[piece[0]][piece[1]]])
                elif row == 0 and col == board_dim - 1 and piece[0] == "F":
                    print("elif 3")
                    possible_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_first_row_last_col[piece[1]]])
                elif row != 0 and row != board_dim - 1 and col == 0:
                    print("elif 4")
                    possible_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_not_first_last_row_first_col[piece[0]][piece[1]]])
                elif row == board_dim - 1 and col == 0 and piece[0] == "F":
                    print("elif 5")
                    possible_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_last_row_first_col[piece[1]]])
                elif row != 0 and row != board_dim - 1 and col == board_dim - 1:
                    print("elif 6")
                    possible_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_not_first_last_row_last_col[piece[0]][piece[1]]])
                elif row == board_dim - 1 and col != 0 and col != board_dim - 1:
                    possible_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_last_row_not_first_last_col[piece[0]][piece[1]]])
                elif row == board_dim - 1 and col == board_dim - 1 and piece[0] == "F":
                    print("elif 7")
                    possible_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_last_row_last_col[piece[1]]])
                else:
                    print("else")
                    possible_actions.extend([(row, col, rotation) for rotation in range(1, 4)])  # Add 3 (all) possible rotations

        return possible_actions


    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de PipeMania."""

    def __init__(self, grid):
        """Cria um objeto Board que contém uma grelha n x n."""
        self.grid = grid

    def copy(self):
        """Cria uma cópia do tabuleiro."""
        return Board([row[:] for row in self.grid])

    def is_grid_index(self, row: int, col: int) -> bool:
        """Devolve True se a posição do tabuleiro é válida, False caso contrário."""
        return 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0])

    def set_value(self, row: int, col: int, value: str):
        """Define o valor na posição especificada do tabuleiro."""
        if self.is_grid_index(row, col):
            self.grid[row][col] = value

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if self.is_grid_index(row, col):
            return self.grid[row][col]
        else:
            return None
    
    def get_water_pipes(self, piece: str):
        """Retorna o vetor que representa as saídas de água da peça especificada."""

        water_pipes = {
            # [Cima, Direita, Baixo, Esquerda]
            "FC": [1, 0, 0, 0],
            "FB": [0, 0, 1, 0],
            "FE": [0, 0, 0, 1],
            "FD": [0, 1, 0, 0],
            "BC": [1, 1, 0, 1],
            "BB": [0, 1, 1, 1],
            "BE": [1, 0, 1, 1],
            "BD": [1, 1, 1, 0],
            "VC": [1, 0, 0, 1],
            "VB": [0, 1, 1, 0],
            "VE": [0, 0, 1, 1],
            "VD": [1, 1, 0, 0],
            "LH": [0, 1, 0, 1],
            "LV": [1, 0, 1, 0],
        }

        if piece is None:
            return None

        piece = piece.upper()

        return water_pipes.get(piece)
    
    def compare_piece_connections(self, current_piece: List[int], left_piece: List[int], piece_above: List[int]) -> bool:
        """Compara as saídas de água da peça especificada com as peças que estão à sua esquerda e em cima"""

        if left_piece is None and piece_above is None:
            return True

        elif piece_above is None and left_piece is not None:
            if current_piece[3] == left_piece[1]:
                return True
            else:
                return False

        elif left_piece is None and piece_above is not None:
            if current_piece[0] == piece_above[2]:
                return True
            else:
                return False
            
        else:
            if current_piece[0] == piece_above[2] and current_piece[3] == left_piece[1]:
                return True
            else:
                return False

    def make_piece_permanent(self, row, col):
        """Torna a configuração da peça permanente no determinado tabuleiro."""
        piece = self.get_value(row, col)
        self.set_value(row, col, piece.lower())  # Converts the piece to lowercase to signal the permanent spot

    def is_permanent(self, row, col):
        """Verifica se a peça na determinada posição já está na sua configuração permanente."""
        piece = self.get_value(row, col)
        return piece.islower()

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


    def calculate_rotation(self, start_orient: str, end_orient: str) -> int:
        # Mudar comentário para português
        """Calcula o número de rotações no sentido anti-horário necessárias para 
        obter a orientação final end_orient a partir da orientação inicial star_orient"""
        """Calculate the number of anticlockwise rotations needed to transition from start_orient to end_orient."""
        if start_orient in {'H', 'V'}:
            orientations = ['H', 'V']
        else:
            orientations = ['C', 'D', 'B', 'E']
        start_index = orientations.index(start_orient)
        end_index = orientations.index(end_orient)
        anticlockwise_rotations = (start_index - end_index) % len(orientations)
        return anticlockwise_rotations
    
    def rotate_piece_to_config(self, row: int, col: int, desired_config: str):
        """Rotaciona a peça na posição (row, col) até que sua configuração seja igual à configuração desejada."""
        piece = self.get_value(row, col)
        current_orientation = piece[1]
        desired_orientation = desired_config[1]

        while current_orientation != desired_orientation:       #MUDAR
            #Roda a peça para a esquerda até chegar à posição desejada
            self.turn_left(row, col)
            piece = self.get_value(row, col)
            current_orientation = piece[1]

    def rotate_piece(self, row: int, col: int, rotation: int):
        """Roda a peça na determinada posição com base no valor da rotação."""

        while rotation != 0:
            self.turn_left(row, col)
            rotation -= 1
        


    def turn_left(self, row: int, col: int):
        """Roda a peça na posição especificada para a esquerda (sentido anti-horário)."""
        piece = self.get_value(row, col)
        orientations = {'C': 'E', 'E': 'B', 'B': 'D', 'D': 'C', 'H': 'V', 'V': 'H'}
        new_orientation = orientations[piece[1]]
        self.set_value(row, col, piece[0] + new_orientation)

    
    def print_grid(self):
        for row in self.grid:
            print(' '.join(row))

    # TODO: outros metodos da classe


class PipeMania(Problem):

    # Pomos @Override???

    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(initial=PipeManiaState(board))
        # TODO
        pass

    def get_initial_state(self):
        """Devolve o estado inicial."""
        return self.initial

    """

    def change_borders(self):
        Faz uma interpretação do estado atual do tabuleiro e faz alterações no limite do tabuleiro.
        
        initial_board = self.get_initial_state().get_board()
        board_dim = len(initial_board.grid)

        # Define mappings for each border piece configuration
        border_mappings = {
            # First row mappings
            (0, col): {
                ("V", "B"): "VB",
                ("B", "B"): "BB",
                ("L", "H"): "LH",
                ("V", "E"): "VE",
                ("F", "V"): "VF" if col == 0 else None
            },
            # Last row mappings
            (board_dim-1, col): {
                ("V", "D"): "VD",
                ("B", "C"): "BC",
                ("L", "V"): "LV",
                ("F", "V"): "VF" if col == 0 or col == board_dim - 1 else None
            },
            # First column mappings (excluding corners)
            (row, 0): {
                ("L", "V"): "LV",
                ("B", "D"): "BD",
                ("F", "V"): "VF" if row == 0 or row == board_dim - 1 else None
            },
            # Last column mappings (excluding corners)
            (row, board_dim-1): {
                ("B", "E"): "BE",
                ("F", "V"): "VF" if row == 0 or row == board_dim - 1 else None
            }
        }

        # Apply mappings to the borders
        for (row, col), mappings in border_mappings.items():
            piece = initial_board.get_value(row, col)
            new_config = mappings.get((piece[0], piece[1]))
            if new_config:
                initial_board.rotate_piece_to_config(row, col, new_config)
                initial_board.make_piece_permanent(row, col)

    """


    
    def change_borders(self):
        """Faz uma interpretação do estado atual do tabuleiro e faz alterações no limite do tabuleiro."""
        
        initial_board = self.get_initial_state().get_board()
        board_dim = len(initial_board.grid)

        # Percorrer a primeira linha do tabuleiro
        for col in range(board_dim):

            piece = initial_board.get_value(0, col)
    
            if col == 0 and piece[0] == "V":
                if piece[1] != "B":
                    initial_board.rotate_piece_to_config(0, col, "VB")
                initial_board.make_piece_permanent(0, col)

            if piece[0] == "B":
                if piece[1] != "B":
                    initial_board.rotate_piece_to_config(0, col, "BB")
                initial_board.make_piece_permanent(0, col)

            if piece[0] == "L":
                if piece[1] != "H":
                    initial_board.rotate_piece_to_config(0, col, "LH")
                initial_board.make_piece_permanent(0, col)

            if col == board_dim - 1 and piece[0] == "V":
                if piece[1] != "E":
                    initial_board.rotate_piece_to_config(0, col, "VE")
                initial_board.make_piece_permanent(0, col)

        # Percorrer a última linha do tabuleiro
        for col in range(board_dim):

            piece = initial_board.get_value(board_dim-1, col)
    
            if col == 0 and piece[0] == "V":
                if piece[1] != "D":
                    initial_board.rotate_piece_to_config(board_dim-1, col, "VD")
                initial_board.make_piece_permanent(board_dim-1, col)
            
            if piece[0] == "B":
                if piece[1] != "C":
                    initial_board.rotate_piece_to_config(board_dim-1, col, "BC")
                initial_board.make_piece_permanent(board_dim-1, col)

            if piece[0] == "L":
                if piece[1] != "H":
                    initial_board.rotate_piece_to_config(board_dim-1, col, "LH")
                initial_board.make_piece_permanent(board_dim-1, col)

            if col == board_dim - 1 and piece[0] == "V":
                if piece[1] != "C":
                    initial_board.rotate_piece_to_config(board_dim-1, col, "VC")
                initial_board.make_piece_permanent(board_dim-1, col)
        
        # Percorrer a primeira coluna do tabuleiro
        for row in range(board_dim):

            piece = initial_board.get_value(row, 0)

            if piece[0] == "L":
                if piece[1] != "V":
                    initial_board.rotate_piece_to_config(row, 0, "LV")
                initial_board.make_piece_permanent(row, 0)

            if piece[0] == "B":
                if piece[1] != "D":
                    initial_board.rotate_piece_to_config(row, 0, "BD")
                initial_board.make_piece_permanent(row, 0)

        # Percorrer a última coluna do tabuleiro
        for row in range(board_dim):

            piece = initial_board.get_value(row, board_dim-1)

            if piece[0] == "L":
                if piece[1] != "V":
                    initial_board.rotate_piece_to_config(row, board_dim-1, "LV")
                initial_board.make_piece_permanent(row, col)

            if piece[0] == "B":
                if piece[1] != "E":
                    initial_board.rotate_piece_to_config(row, board_dim-1, "BE")
                initial_board.make_piece_permanent(row, col)

    
    



    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

        return state.generate_actions()

        # TODO
        

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        #if action in self.actions(state):   #MAYBE?

        current_board = state.get_board()
        row, col, rotation = action
        new_board = current_board.copy()
        new_board.rotate_piece(row, col, rotation)

        new_state = PipeManiaState(new_board)
        
        return new_state

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""

        board = state.get_board()
        board_dim = len(board.grid)

        for row in range(board_dim):
            for col in range(1, board_dim):
                current_piece = board.get_value(row, col)
                left_piece = board.get_value(row, col - 1)
                above_piece = board.get_value(row - 1, col)

                if not board.compare_piece_connections(board.get_water_pipes(current_piece), board.get_water_pipes(left_piece), board.get_water_pipes(above_piece)):
                    return False
        
        return True





    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:

    board = Board.parse_instance()

    problem = PipeMania(board)
    s0 = PipeManiaState(board)

    problem.change_borders()
    print(problem.actions(s0))
    board.print_grid()

    s1 = problem.result(s0, (0, 1, 3))
    s2 = problem.result(s1, (0, 1, 3))
    s6 = problem.result(s2, (1, 1, 3))
    s7 = problem.result(s6, (2, 0, 1)) # anti-clockwise (exemplo de uso)
    s8 = problem.result(s7, (2, 0, 1)) # anti-clockwise (exemplo de uso)
    s9 = problem.result(s8, (2, 1, 3))
    s10 = problem.result(s9, (2, 1, 3))
    s11 = problem.result(s10, (2, 2, 3))

    print("Is goal?", problem.goal_test(s1))
    print("Is goal?", problem.goal_test(s11))
    print("Solution:\n", s11.get_board().print_grid(), sep="")
    



    

    

    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
