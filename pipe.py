# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 57:
# 106157 Leonor Costa Figueira
# 106322 Raquel dos Anjos Santos Caldeira Rodrigues

nr_voltas = 0

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

from typing import List, Tuple


class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def print_edges(self):
        """Prints the edges of the graph."""
        print(len(self.adjacency_list))
        for vertex, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                print(f"Edge: ({vertex}, {neighbor})")


    def add_edge(self, u, v):
        if u not in self.adjacency_list:
            self.adjacency_list[u] = []
        if v not in self.adjacency_list:
            self.adjacency_list[v] = []
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)

    def dfs(self, node, visited):
        visited.add(node)
        for neighbor in self.adjacency_list.get(node, []):
            if neighbor not in visited:
                self.dfs(neighbor, visited)

    def connected_components(self):
        visited = set()
        subgraphs = []
        for node in self.adjacency_list:
            if node not in visited:
                subgraph = set()
                self.dfs(node, subgraph)
                subgraphs.append(subgraph)
                visited.update(subgraph)
        return subgraphs

    def subgraph_count(self):
        return len(self.connected_components())



class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def get_board(self):
        """Retorna o tabuleiro do estado."""
        return self.board

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        """Método para verificar se dois estados são iguais."""
        return isinstance(other, PipeManiaState) and self.board == other.board

    def check_neighbors(self, unfiltered_actions: dict):
        """"""

        print("unfiltered_actions:")
        for piece_key in unfiltered_actions:
            print("piece_key:", piece_key)
            print("value:", unfiltered_actions[piece_key])

        board = self.get_board()
        board_dim = len(board.grid)
        all_actions = []
        actions_to_remove = []
        permanent_pieces = 0

        for row in range(board_dim):
            for col in range(board_dim):
                if board.is_permanent(row, col):
                    permanent_pieces += 1
        
        not_permanent_pieces = board_dim * board_dim - permanent_pieces


        for row in range(board_dim):
            for col in range(board_dim):

                if board.is_permanent(row, col):
                    continue

                current_key = (row, col)
                current_actions = []

                # Define os índices das peças vizinhas
                neighbor = [
                    (row - 1, col),  # Cima
                    (row, col + 1),  # Direita
                    (row + 1, col),  # Baixo
                    (row, col - 1),  # Esquerda
                ]
                
                # Vetor com saídas de água da peça atual que será
                # construído através das peças vizinhas já permanentes
                piece_water_pipes = [-1] * 4

                piece = board.get_value(row, col)

                for i in range(0, 4):
                    neighbor_row, neighbor_col = neighbor[i]

                    if board.is_grid_index(neighbor_row, neighbor_col):
                        neighbor_piece = board.get_value(neighbor_row, neighbor_col)
                        neighbor_permanent = board.is_permanent(neighbor_row, neighbor_col)
                        
                        if neighbor_permanent:
                                                                                        #Calcular o índice da saída de água da peça vizinha
                            piece_water_pipes[i] = board.get_water_pipes(neighbor_piece)[(i + 2) % len(neighbor)]

                possible_configs = board.find_matching_pieces(piece, piece_water_pipes)

                for rotated_piece in possible_configs:
                    rotation = self.board.calculate_rotation(piece[1], rotated_piece[1])
                    #if rotation:    # Se a rotação der 0, a peça já está na posição correta e a ação não é adicionada
                    current_actions.extend([(row, col, rotation)])
                
                if row == 1 and col == 2:
                    print("current_actions da (1,2):")
                    print(current_actions)
                
                if current_key in unfiltered_actions:
                    unfiltered_actions[current_key].append(current_actions)
                else:
                    unfiltered_actions[current_key] = [current_actions]

                print("two lists:")
                for piece_key in unfiltered_actions:
                    print("piece_key:", piece_key)
                    print("value:", unfiltered_actions[piece_key])

                if len(unfiltered_actions[current_key]) == 1:
                    unfiltered_actions[current_key] = unfiltered_actions[current_key][0]
                elif len(unfiltered_actions[current_key]) == 2:
                    unfiltered_actions[current_key] = list(set(unfiltered_actions[current_key][0]).intersection(unfiltered_actions[current_key][1]))

                # Se apenas houver uma ação possível tornamos a peça permanente
                #if len(unfiltered_actions[current_key]) in {0,1}:
                if len(unfiltered_actions[current_key]) == 1:
                    board.rotate_piece(current_key[0], current_key[1], unfiltered_actions[current_key][0][2])
                    actions_to_remove.append(current_key)
                    board.make_piece_permanent(current_key[0], current_key[1])

                    # Quando se colocou permanente a última peça que faltava 
                    if not_permanent_pieces == 1:
                        all_actions.append((0, 0, 0))
                    not_permanent_pieces -= 1


        # Remove as ações das peças que se tornaram permanentes
        for piece_key in actions_to_remove:
            unfiltered_actions.pop(piece_key)

        for actions in unfiltered_actions.values():
            for action in actions:
                # Descartar as ações que mantêm as peças com a mesma orientação
                if action[2] != 0:
                    all_actions.append(action)

        
        print("after intersection:")
        for piece_key in unfiltered_actions:
            print("piece_key:", piece_key)
            print("value:", unfiltered_actions[piece_key])
            
        #print(all_actions)

        return [(0,4,0)]
        #return all_actions


    def update_neighbors(self):
        """"""
        
        board = self.get_board()
        board_dim = len(board.grid)

        


    def generate_actions(self):
        """Faz uma interpretação do estado atual do tabuleiro e gera ações possíveis."""
        possible_actions = {}
        board = self.get_board()
        board_dim = len(board.grid)

        # Canto superior esquerdo
        rotation_mappings_first_row_first_col = {'C': ['B', 'D'], 'E': ['B', 'D'], 'B': ['B', 'D'], 'D': ['B', 'D']}

        # Primeira linha
        rotation_mappings_first_row_not_first_last_col = {
            'F': {'C': ['B', 'E', 'D'], 'E': ['B', 'E', 'D'], 'B': ['B', 'E', 'D'], 'D': ['B', 'E', 'D']},
            'V': {'C': ['B', 'E'], 'E': ['B', 'E'], 'B': ['B', 'E'], 'D': ['B', 'E']}
        }

        # Canto superior direito
        rotation_mappings_first_row_last_col = {'C': ['B', 'E'], 'E': ['B', 'E'], 'B': ['B', 'E'], 'D': ['B', 'E']}

        # Primeira coluna
        rotation_mappings_not_first_last_row_first_col = {
            'F': {'C': ['B', 'C', 'D'], 'E': ['B', 'C', 'D'], 'B': ['B', 'C', 'D'], 'D': ['B', 'C', 'D']},
            'V': {'C': ['B', 'D'], 'E': ['B', 'D'], 'B': ['B', 'D'], 'D': ['B', 'D']}
        }

        # Canto inferior esquerdo
        rotation_mappings_last_row_first_col = {'C': ['C', 'D'], 'E': ['C', 'D'], 'B': ['C', 'D'], 'D': ['C', 'D']}

        # Última coluna
        rotation_mappings_not_first_last_row_last_col = {
            'F': {'C': ['B', 'C', 'E'], 'E': ['B', 'C', 'E'], 'B': ['B', 'C', 'E'], 'D': ['B', 'C', 'E']},
            'V': {'C': ['C', 'E'], 'E': ['C', 'E'], 'B': ['C', 'E'], 'D': ['C', 'E']}
        }

        # Última linha
        rotation_mappings_last_row_not_first_last_col = {
            'F': {'C': ['C', 'E', 'D'], 'E': ['C', 'E', 'D'], 'B': ['C', 'E', 'D'], 'D': ['C', 'E', 'D']},
            'V': {'C': ['C', 'D'], 'E': ['C', 'D'], 'B': ['C', 'D'], 'D': ['C', 'D']}
        }

        # Canto inferior direito
        rotation_mappings_last_row_last_col = {'C': ['C', 'E'], 'E': ['C', 'E'], 'B': ['C', 'E'], 'D': ['C', 'E']}
        
        for row in range(len(self.board.grid)):
            for col in range(len(self.board.grid[0])):

                piece = self.board.get_value(row, col)
                current_key = (row, col)
                current_actions = []
                #print("Coordenadas: (", row, ",", col, ") Peça atual: ", piece)

                if board.is_permanent(row, col):
                    pass
                elif row == 0 and col == 0 and piece[0] == "F":
                    #print("elif 1")
                    current_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_first_row_first_col[piece[1]]])
                elif row == 0 and col != 0 and col != board_dim - 1:
                    #print("elif 2")
                    current_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_first_row_not_first_last_col[piece[0]][piece[1]]])
                elif row == 0 and col == board_dim - 1 and piece[0] == "F":
                    #print("elif 3")
                    current_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_first_row_last_col[piece[1]]])
                elif row != 0 and row != board_dim - 1 and col == 0:
                    #print("elif 4")
                    current_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_not_first_last_row_first_col[piece[0]][piece[1]]])
                elif row == board_dim - 1 and col == 0 and piece[0] == "F":
                    #print("elif 5")
                    current_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_last_row_first_col[piece[1]]])
                elif row != 0 and row != board_dim - 1 and col == board_dim - 1:
                    #print("elif 6")
                    current_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_not_first_last_row_last_col[piece[0]][piece[1]]])
                elif row == board_dim - 1 and col != 0 and col != board_dim - 1:
                    current_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_last_row_not_first_last_col[piece[0]][piece[1]]])
                elif row == board_dim - 1 and col == board_dim - 1 and piece[0] == "F":
                    #print("elif 7")
                    current_actions.extend([(row, col, self.board.calculate_rotation(piece[1], new_orientation))
                                            for new_orientation in rotation_mappings_last_row_last_col[piece[1]]])
                else:
                    #print("else")
                    # LIMITAR PEÇAS L...
                    current_actions.extend([(row, col, rotation) for rotation in range(0, 4)])  # Add 3 (all) possible rotations

                if current_actions:
                    possible_actions[current_key] = [current_actions]

        return self.check_neighbors(possible_actions)



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

    def adjacent_vertical_values(self, row: int, col: int) -> Tuple[str,str]:
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        if self.is_grid_index(row, col):
            return self.get_value(row - 1, col), self.get_value(row + 1, col)
        else:
            return None # ou temos de pôr (None, None) ? caso a peça não exista

    def adjacent_horizontal_values(self, row: int, col: int) -> Tuple[str,str]:
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

    def get_pipes(self, piece_type: chr) -> dict:
        """Retorna o dicionário das saídas de água da peça especificada."""

        water_pipes = {
            # [Cima, Direita, Baixo, Esquerda]
            "F": {"FC": [1, 0, 0, 0],
                  "FB": [0, 0, 1, 0],
                  "FE": [0, 0, 0, 1],
                  "FD": [0, 1, 0, 0]},

            "B": {"BC": [1, 1, 0, 1],
                  "BB": [0, 1, 1, 1],
                  "BE": [1, 0, 1, 1],
                  "BD": [1, 1, 1, 0]},
                
            "V": {"VC": [1, 0, 0, 1],
                  "VB": [0, 1, 1, 0],
                  "VE": [0, 0, 1, 1],
                  "VD": [1, 1, 0, 0]},
            
            "L": {"LH": [0, 1, 0, 1],
                  "LV": [1, 0, 1, 0]}
        }

        return water_pipes[piece_type]


    def get_water_pipes(self, piece: str) -> List[int]:
        """Retorna o vetor que representa as saídas de água da peça especificada."""

        if piece is None:
            return None

        piece = piece.upper()

        return self.get_pipes(piece[0]).get(piece)
    

    def find_matching_pieces(self, current_piece: str, piece_water_pipes: List[int]):
        """"""
        
        #print(self.get_pipes(current_piece[0]))
        water_pipes_dict = self.get_pipes(current_piece[0])

        # Cria uma cópia das chaves para mais tarde iterar sobre as mesmas
        pieces_to_remove = []

        for piece, water_pipes in water_pipes_dict.items():
            #print(current_piece)
            # print(piece_water_pipes)
            # print(water_pipes)
            for i in range(0,4):
                if piece_water_pipes[i] == -1:
                    continue
                if water_pipes[i] != piece_water_pipes[i]:
                    pieces_to_remove.append(piece)
                    break

        # print(pieces_to_remove)
        # print(water_pipes_dict.keys())


        # Remove as peças que devem ser retiradas
        for piece in pieces_to_remove:
            water_pipes_dict.pop(piece)

        #print(water_pipes_dict.keys())

        return list(water_pipes_dict.keys())


    
    def check_connections(self, current_piece: List[int], other_piece: List[int], comparison: int) -> int:
        """Compara as saídas de água da peça especificada com a peça à esquerda se left_comparison for True,
        caso contrário compara com a peça acima. 
        1 -> Peça esquerda
        2 -> Peça cima
        3 -> Peça direita
        4 -> Peça baixo
        Devolve 0 caso as duas peças sejam compatíveis sem ligação,
        1 caso forem compatíveis com ligação, 2 se forem incompatíveis."""

        if comparison == 1: # Compara com a peça à esquerda
            if current_piece[3] == other_piece[1] == 0:
                return 0
            elif current_piece[3] == other_piece[1] == 1:
                return 1

        elif comparison == 2: # Compara com a peça acima
            if current_piece[0] == other_piece[2] == 0:
                return 0
            elif current_piece[0] == other_piece[2] == 1:
                return 1

        elif comparison == 3: # Compara com a peça à direita
            if current_piece[1] == other_piece[3] == 0:
                return 0
            elif current_piece[1] == other_piece[3] == 1:
                return 1

        elif comparison == 4: # Compara com a peça de baixo
            if current_piece[2] == other_piece[0] == 0:
                return 0
            elif current_piece[2] == other_piece[0] == 1:
                return 1

        return 2


    
    
    def compare_piece_connections(self, current_piece: List[int], left_piece: List[int], above_piece: List[int]) -> bool:
        """Compara as saídas de água da peça especificada com as peças que estão
        à sua esquerda e em cima. Devolve True se forem compatíveis, False caso contrário."""
            
        if left_piece is None and above_piece is None: # canto superior esquerdo
            return True

        elif above_piece is None and left_piece is not None: # 1ª linha
            # Esquerda da peça atual VS Direita da peça à esquerda
            if self.check_connections(current_piece, left_piece, 1) != 2:
                return True
            else:
                return False

        elif left_piece is None and above_piece is not None: # 1ª coluna
            # Cima da peça atual VS Baixo da peça acima
            if self.check_connections(current_piece, above_piece, 2) != 2:
                return True
            else:
                return False

        else:                                                # resto do tabuleiro
            # Combina as duas comparações anteriores
            if self.check_connections(current_piece, above_piece, 2) != 2 and \
                self.check_connections(current_piece, left_piece, 1) != 2:
                return True
            else:
                return False
               

    def make_piece_permanent(self, row, col):
        """Torna a configuração da peça permanente no determinado tabuleiro."""
        piece = self.get_value(row, col)
        self.set_value(row, col, piece.lower())  # Altera a peça para letras minúsculas 
                                                 # para assinalar que está permanente

    def is_permanent(self, row, col):
        """Verifica se a peça na determinada posição já está na sua configuração permanente."""
        piece = self.get_value(row, col)
        return piece.islower()

    


    def calculate_rotation(self, start_orient: str, end_orient: str) -> int:
        """Calcula o número de rotações no sentido anti-horário necessárias para 
        obter a orientação final end_orient a partir da orientação inicial star_orient"""
        if start_orient in {'H', 'V'}:
            orientations = ['H', 'V']
        else:
            orientations = ['C', 'D', 'B', 'E']
        start_index = orientations.index(start_orient)
        end_index = orientations.index(end_orient)
        anticlockwise_rotations = (start_index - end_index) % len(orientations)
        return anticlockwise_rotations
    
    def rotate_piece_to_config(self, row: int, col: int, desired_config: str):
        """Rotaciona a peça na posição (row, col) para a configuração desejada."""
        piece = self.get_value(row, col)
        new_piece = piece[0] + desired_config[1] # Creates a new piece with the desired configuration
        self.set_value(row, col, new_piece)         


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

    
    def print_grid_debug(self):
        for row in self.grid:
            print(' '.join(row))


    def print_grid(self):
        for row in self.grid:
            # Converts each piece to uppercase before printing
            print('\t'.join(piece.upper() for piece in row))


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
        new_board.make_piece_permanent(row, col)

        new_state = PipeManiaState(new_board)
        
        return new_state

    def goal_test(self, state: PipeManiaState):
        global nr_voltas
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""

        board = state.get_board()
        board_dim = len(board.grid)
        #board.print_grid_debug()
        if nr_voltas == 0:
            board.print_grid_debug()
        if nr_voltas == 1:
            board.print_grid_debug()
            exit()
        nr_voltas += 1

        for row in range(board_dim):
            for col in range(board_dim):
                current_piece = board.get_value(row, col)
                left_piece = board.get_value(row, col - 1)
                above_piece = board.get_value(row - 1, col)

                if not board.compare_piece_connections(board.get_water_pipes(current_piece), 
                                                       board.get_water_pipes(left_piece), 
                                                       board.get_water_pipes(above_piece)):
                    return False
        
        # Creates a graph to check if there are subsets of water connected pipes
        graph = Graph()

        for row in range(board_dim):
            for col in range(board_dim):
                current_piece = board.get_value(row, col)
                left_piece = board.get_value(row, col - 1)
                above_piece = board.get_value(row - 1, col)

                node_id = row * board_dim + col
                left_neighbor_id = row * board_dim + col - 1
                above_neighbor_id = (row - 1) * board_dim + col

                if left_piece is not None and (board.check_connections(board.get_water_pipes(current_piece),
                                                                      board.get_water_pipes(left_piece), 1) == 1):
                    graph.add_edge(node_id, left_neighbor_id)
                
                if above_piece is not None and (board.check_connections(board.get_water_pipes(current_piece),
                                                                       board.get_water_pipes(above_piece), 2) == 1):
                    graph.add_edge(node_id, above_neighbor_id)
        

        if graph.subgraph_count() > 1:
            #print("graph.subgraph_count")
            #print(graph.subgraph_count())
            return False

        return True





    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":

    
    board = Board.parse_instance()
    problem = PipeMania(board)
    problem.change_borders()
    #board.print_grid_debug()
    # Obter o nó solução usando a procura em profundidade:
    goal_node = depth_first_tree_search(problem)
    # Verificar se foi atingida a solução
    goal_node.state.get_board().print_grid()
    


    
    # board = Board.parse_instance()
    # problem = PipeMania(board)
    # s0 = PipeManiaState(board)
    # problem.change_borders()
    # print(problem.actions(s0))
    # board.print_grid_debug()
    # print("Is goal?", problem.goal_test(s0))
    # print("Solution:")
    # s0.get_board().print_grid()
    

    

    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
