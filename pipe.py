# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 57:
# 106157 Leonor Costa Figueira
# 106322 Raquel dos Anjos Santos Caldeira Rodrigues

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
    """Classe para representar um grafo com os seus nós e arcos. 
    Os arcos são armazenados numa lista de adjacências."""

    def __init__(self):
        """Cria um grafo e inicializa a lista de adjacências."""
        self.adjacency_list = {}

    def print_edges(self):
        """Imprime os arcos do grafo."""
        print(len(self.adjacency_list))
        for vertex, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                print(f"Edge: ({vertex}, {neighbor})")

    def add_edge(self, u, v):
        """Adiciona um arco entre dois nós ao grafo."""
        if u not in self.adjacency_list:
            self.adjacency_list[u] = []
        if v not in self.adjacency_list:
            self.adjacency_list[v] = []
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)

    def iterative_DFS(self, s, subgraph, visited):
        """DFS iterativa para encontrar um subgrafo a partir do nó s."""
        stack = [s]

        while stack:
            s = stack.pop()
            if s not in visited:
                subgraph.add(s)  # Adiciona o nó ao subgrafo
                visited.add(s)   # Assinala o nó como visitado
                stack.extend(self.adjacency_list[s][::-1])

    def connected_components(self):
        """Encontra e retorna os componentes fortemente ligados do grafo."""
        visited = set()
        subgraphs = []
        for node in self.adjacency_list:
            if node not in visited:
                subgraph = set()
                self.iterative_DFS(node, subgraph, visited)
                subgraphs.append(subgraph)
        return subgraphs

    def subgraph_count(self):
        """Retorna o número de subgrafos independentes do grafo."""
        return len(self.connected_components())



class PipeManiaState:
    """Classe que armazena um estado do problema PipeMania."""
    state_id = 0

    def __init__(self, board):
        """Cria um estado pipemania, guardando o board correspondente."""
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
    
    def check_neighbors(self, unfiltered_actions: dict) -> List[Tuple[int]]:
        """Método para percorrer o tabuleiro deste estado e devolver possíveis 
        ações a executar tendo em conta a lista pré-determinada de ações 
        (unfiltered_actions) e restrições causadas pelas peças vizinhas.
        À medida que o tabuleiro é percorrido tornam-se imediatamente permanentes
        as peças que têm apenas uma configuração possível."""

        board = self.get_board()
        board_dim = len(board.grid)
        actions_to_remove = []
        permanent_pieces = 0

        for row in range(board_dim):
            for col in range(board_dim):
                if board.is_permanent(row, col):
                    permanent_pieces += 1
        
        # Calcula o número de peças que faltam tornar permanentes
        not_permanent_pieces = board_dim * board_dim - permanent_pieces

        for row in range(board_dim):
            for col in range(board_dim):

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

                # Se a peça é permanente, verificar as peças vizinhas
                if board.is_permanent(row, col):
                    
                    for i in range(0, 4):
                        neighbor_row, neighbor_col = neighbor[i]

                        if board.is_grid_index(neighbor_row, neighbor_col):
                            neighbor_piece = board.get_value(neighbor_row, neighbor_col)
                            neighbor_permanent = board.is_permanent(neighbor_row, neighbor_col)
                            
                            if neighbor_permanent:
                                                                                            #Calcular o índice da saída de água da peça vizinha
                                piece_water_pipes[i] = board.get_water_pipes(neighbor_piece)[(i + 2) % len(neighbor)]

                    piece = piece.upper()
                    possible_configs = board.find_matching_pieces(piece, piece_water_pipes)

                    # Significa que o estado tem um tabuleiro errado,
                    # existem peças vizinhas permanentes incompatíveis
                    if len(possible_configs) == 0:
                        # O ramo do nó com este estado deve ser descartado
                        return []
                    
                    continue

                current_key = (row, col)
                current_actions = []

                for i in range(0, 4):
                    neighbor_row, neighbor_col = neighbor[i]

                    if board.is_grid_index(neighbor_row, neighbor_col):
                        neighbor_piece = board.get_value(neighbor_row, neighbor_col)
                        neighbor_permanent = board.is_permanent(neighbor_row, neighbor_col)
                        
                        if neighbor_permanent:
                            piece_water_pipes[i] = board.get_water_pipes(neighbor_piece)[(i + 2) % len(neighbor)]

                possible_configs = board.find_matching_pieces(piece, piece_water_pipes)

                # Significa que o estado tem um tabuleiro errado,
                # a peça não tem qualquer configuração possível
                if len(possible_configs) == 0:
                    # O ramo do nó com este estado deve ser descartado
                    return []

                # Adiciona as ações possíveis para a peça dada
                for rotated_piece in possible_configs:
                    rotation = self.board.calculate_rotation(piece[1], rotated_piece[1])
                    current_actions.extend([(row, col, rotation)])
                
                # Adiciona as ações ao dicionário
                if current_key in unfiltered_actions:
                    unfiltered_actions[current_key].append(current_actions)
                else:
                    unfiltered_actions[current_key] = [current_actions]

                if len(unfiltered_actions[current_key]) == 1:
                    # Obtem a única lista de ações da peça dada
                    unfiltered_actions[current_key] = unfiltered_actions[current_key][0]
                elif len(unfiltered_actions[current_key]) == 2:
                    # Faz a interseção entre as duas listas de ações para a peça dada
                    unfiltered_actions[current_key] = list(
                        set(unfiltered_actions[current_key][0]).intersection(unfiltered_actions[current_key][1]))

                # Se apenas houver uma ação possível tornamos a peça permanente
                if len(unfiltered_actions[current_key]) == 1:
                    board.rotate_piece(current_key[0], current_key[1], unfiltered_actions[current_key][0][2])
                    actions_to_remove.append(current_key)
                    board.make_piece_permanent(current_key[0], current_key[1])

                    # Quando se colocou permanente a última peça que faltava 
                    if not_permanent_pieces == 1:
                        return [(0, 0, 0)]
                    not_permanent_pieces -= 1


        # Remove as ações das peças que se tornaram permanentes
        for piece_key in actions_to_remove:
            unfiltered_actions.pop(piece_key)

        for value in unfiltered_actions.values():
            # Retorna o valor da primeira chave existente no dicionário
            return value
        
        # Caso que o tabuleiro tem as peças todas permanentes mas não é solução,
        # o ramo do nó com este estado deve ser descartado
        return []
    


    def get_orientations(self, row: int, col: int, piece_type: chr) -> List[chr]:
        """Retorna as orientações possíveis da peça dada."""

        corner_orientations = {
            'top_left': {'F': ['B', 'D']},
            'top_right': {'F': ['B', 'E']},
            'bottom_left': {'F': ['C', 'D']},
            'bottom_right': {'F': ['C', 'E']}
        }
    
        border_orientations = {
            'top': {'F': ['B', 'E', 'D'], 'V': ['B', 'E']},
            'bottom': {'F': ['C', 'E', 'D'], 'V': ['C', 'D']},
            'left': {'F': ['B', 'C', 'D'], 'V': ['B', 'D']},
            'right': {'F': ['B', 'C', 'E'], 'V': ['C', 'E']}
        }

        board_dim = len(self.get_board().grid)

        if piece_type == 'L':
            return ['H', 'V']
        elif row > 0 and col > 0 and row < board_dim-1 and col < board_dim-1:
            return ['C', 'D', 'B', 'E']

        elif row == 0:
            if col == 0:
                return corner_orientations['top_left'].get(piece_type)
            elif col == board_dim - 1:
                return corner_orientations['top_right'].get(piece_type)
            else:
                return border_orientations['top'].get(piece_type)
        elif row == board_dim - 1:
            if col == 0:
                return corner_orientations['bottom_left'].get(piece_type)
            elif col == board_dim - 1:
                return corner_orientations['bottom_right'].get(piece_type)
            else:
                return border_orientations['bottom'].get(piece_type)
        elif col == 0:
            return border_orientations['left'].get(piece_type)
        elif col == board_dim - 1:
            return border_orientations['right'].get(piece_type)
        

    def generate_actions(self) -> List[Tuple[int]]:
        """Faz uma interpretação do estado atual do tabuleiro e gera ações possíveis
        tendo em conta os tipos de peças e as suas posições no tabuleiro."""
        possible_actions = {}
        board = self.get_board()
        board_dim = len(board.grid)

        for row in range(board_dim):
            for col in range(board_dim):
                
                piece = self.board.get_value(row, col)
                if not board.is_permanent(row, col):
                    piece_type, current_orientation = piece                    
                    current_key = (row, col)
                    current_actions = []
                    new_orientations = self.get_orientations(row, col, piece_type)
                
                    if new_orientations:
                        current_actions.extend([(row, col, self.get_board().calculate_rotation(current_orientation, new_orientation))
                                                for new_orientation in new_orientations])
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
        """Retorna o dicionário das saídas de água da peça especificada.
        Posição a 1: tem saída de água.
        Posição a 0: não tem saída de água."""

        water_pipes = {
            # [Cima, Direita, Baixo, Esquerda]
            "F": {"FC": [1, 0, 0, 0], # FC só tem saída de água para cima
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
    

    def find_matching_pieces(self, current_piece: str, piece_water_pipes: List[int]) -> List[str]:
        """Devolve as configurações da peça current_piece que
        são compatíveis com as saídas de água em piece_water_pipes."""
        
        # Obtém as configurações possíveis para as saídas de água da current_piece,
        # incluindo as diferentes rotações da peça
        water_pipes_dict = self.get_pipes(current_piece[0])

        # Vetor que guarda as configurações das peças incompatíveis
        pieces_to_remove = []

        for piece, water_pipes in water_pipes_dict.items():
            # Compara as saídas de água dos dois vetores
            for i in range(0,4):
                if piece_water_pipes[i] == -1:  # Saída de água desconhecida
                    continue
                if water_pipes[i] != piece_water_pipes[i]:  # Incompatibilidade encontrada
                    pieces_to_remove.append(piece)
                    break

        # Remove as rotações da peça para as quais foram encontradas incompatibilidades
        for piece in pieces_to_remove:
            water_pipes_dict.pop(piece)

        return list(water_pipes_dict.keys())

    
    def check_connections(self, current_piece: List[int], other_piece: List[int], comparison: int) -> int:
        """Compara as saídas de água da peça especificada com a peça adjacente 
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
            
        if left_piece is None and above_piece is None: # Canto superior esquerdo
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

        else:                                                # Resto do tabuleiro
            # Combina as duas comparações anteriores
            if self.check_connections(current_piece, above_piece, 2) != 2 and \
                self.check_connections(current_piece, left_piece, 1) != 2:
                return True
            else:
                return False
               

    def make_piece_permanent(self, row: int, col: int):
        """Torna a configuração da peça permanente no tabuleiro."""
        piece = self.get_value(row, col)
        self.set_value(row, col, piece.lower())  # Altera a peça para letras minúsculas 
                                                 # para assinalar que está permanente

    def is_permanent(self, row: int, col: int) -> bool:
        """Verifica se a peça na posição dada já está permanente."""
        piece = self.get_value(row, col)
        return piece.islower()

    def calculate_rotation(self, start_orient: str, end_orient: str) -> int:
        """Calcula o número de rotações no sentido anti-horário necessárias para 
        obter a orientação final end_orient a partir da orientação inicial start_orient."""
        if start_orient in {'H', 'V'}:
            orientations = ['H', 'V']
        else:
            orientations = ['C', 'D', 'B', 'E']
        start_index = orientations.index(start_orient)
        end_index = orientations.index(end_orient)
        anticlockwise_rotations = (start_index - end_index) % len(orientations)
        return anticlockwise_rotations
    
    def rotate_piece_to_config(self, row: int, col: int, desired_config: str):
        """Rotaciona a peça na posição dada para a configuração desejada."""
        piece = self.get_value(row, col)
        new_piece = piece[0] + desired_config[1] # Cria uma nova peça com a configuração desejada
        self.set_value(row, col, new_piece)         

    def rotate_piece(self, row: int, col: int, rotation: int):
        """Roda a peça na posição dada com base no valor de rotation."""
        while rotation != 0:
            self.turn_left(row, col)
            rotation -= 1
        
    def turn_left(self, row: int, col: int):
        """Roda a peça na posição especificada para a esquerda 
        (90º no sentido anti-horário)."""
        piece = self.get_value(row, col)
        orientations = {'C': 'E', 'E': 'B', 'B': 'D', 'D': 'C', 'H': 'V', 'V': 'H'}
        new_orientation = orientations[piece[1]]
        self.set_value(row, col, piece[0] + new_orientation)
    
    def print_grid(self):
        """Imprime as nxn peças da grelha deste board."""
        for row in self.grid:
                            # Converte cada peça para maiúsculas antes de imprimir
            print('\t'.join(piece.upper() for piece in row))


class PipeMania(Problem):

    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(initial=PipeManiaState(board))

    def get_initial_state(self):
        """Devolve o estado inicial."""
        return self.initial
    
    def change_borders(self):
        """Faz uma interpretação do estado atual do tabuleiro e faz alterações no limite do tabuleiro."""

        initial_board = self.get_initial_state().get_board()
        board_dim = len(initial_board.grid)

        # Define as corretas configurações para as peças da borda
        piece_border_configs = {
            "V": {"top_left": "VB", "top_right": "VE", "bottom_left": "VD", "bottom_right": "VC"},
            "B": {"left": "BD", "right": "BE", "top": "BB", "bottom": "BC"},
            "L": {"left": "LV", "right": "LV", "top": "LH", "bottom": "LH"},
        }

        for row in range(board_dim):
            for col in range(board_dim):
                piece_type, orientation = initial_board.get_value(row, col)

                # Se a peça for do tipo "fecho" ou se a posição a iterar não está em nenhuma das bordas
                if piece_type == "F" or (row > 0 and col > 0 and row < board_dim-1 and col < board_dim-1):
                    continue

                elif piece_type == "V":
                    if row == 0 and col == 0:
                        border_config = piece_border_configs[piece_type]["top_left"]
                    elif row == board_dim - 1 and col == 0:
                        border_config = piece_border_configs[piece_type]["bottom_left"]
                    elif row == 0 and col == board_dim - 1:
                        border_config = piece_border_configs[piece_type]["top_right"]
                    elif row == board_dim - 1 and col == board_dim - 1:
                        border_config = piece_border_configs[piece_type]["bottom_right"]
                    else:
                        continue

                elif piece_type == "B" or piece_type == "L":
                    if row == 0:
                        border_config = piece_border_configs[piece_type]["top"]
                    elif row == board_dim - 1:
                        border_config = piece_border_configs[piece_type]["bottom"]
                    elif col == 0:
                        border_config = piece_border_configs[piece_type]["left"]
                    elif col == board_dim - 1:
                        border_config = piece_border_configs[piece_type]["right"]
                    else:
                        continue
                
                # Se for necessário alterar a orientação da peça
                if orientation != border_config[1]:
                    initial_board.rotate_piece_to_config(row, col, border_config)
                initial_board.make_piece_permanent(row, col)
        

    def actions(self, state: PipeManiaState) -> List[Tuple[int]]:
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        return state.generate_actions()


    def result(self, state: PipeManiaState, action) -> PipeManiaState:
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        current_board = state.get_board()
        row, col, rotation = action
        
        new_board = current_board.copy()
        new_board.rotate_piece(row, col, rotation)
        new_board.make_piece_permanent(row, col)
        new_state = PipeManiaState(new_board)
        
        return new_state

    def goal_test(self, state: PipeManiaState) -> bool:
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Verifica se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""

        board = state.get_board()
        board_dim = len(board.grid)

        for row in range(board_dim):
            for col in range(board_dim):
                current_piece = board.get_value(row, col)
                left_piece = board.get_value(row, col - 1)
                above_piece = board.get_value(row - 1, col)

                # Verifica que a peça atual tem ligações compatíveis com as peças
                # à esquerda e acima se existirem
                if not board.compare_piece_connections(board.get_water_pipes(current_piece), 
                                                       board.get_water_pipes(left_piece), 
                                                       board.get_water_pipes(above_piece)):
                    return False
        
        # Cria um grafo para verificar que não existem subconjuntos de canalizações
        graph = Graph()

        for row in range(board_dim):
            for col in range(board_dim):
                current_piece = board.get_value(row, col)
                left_piece = board.get_value(row, col - 1)
                above_piece = board.get_value(row - 1, col)

                node_id = row * board_dim + col
                left_neighbor_id = row * board_dim + col - 1
                above_neighbor_id = (row - 1) * board_dim + col

                # Acrescenta um arco ao grafo se existir ligação entre a peça 
                # atual e a peça à esquerda
                if left_piece is not None and (board.check_connections(board.get_water_pipes(current_piece),
                                                                      board.get_water_pipes(left_piece), 1) == 1):
                    graph.add_edge(node_id, left_neighbor_id)
                
                # Acrescenta um arco ao grafo se existir ligação entre a peça 
                # atual e a peça acima
                if above_piece is not None and (board.check_connections(board.get_water_pipes(current_piece),
                                                                       board.get_water_pipes(above_piece), 2) == 1):
                    graph.add_edge(node_id, above_neighbor_id)
        

        if graph.subgraph_count() > 1:  # Caso de rejeição
            return False

        return True
    
if __name__ == "__main__":

    # Lê o ficheiro do standard input
    board = Board.parse_instance()
    problem = PipeMania(board)
    # Faz alterações possíveis nas bordas do tabuleiro
    problem.change_borders()
    # Obtém o nó solução usando a procura em profundidade
    goal_node = depth_first_tree_search(problem)
    # Retira a solução a partir do nó resultante e imprime no formato indicado
    goal_node.state.get_board().print_grid()
