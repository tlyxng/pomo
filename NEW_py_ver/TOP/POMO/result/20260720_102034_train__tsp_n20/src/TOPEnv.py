
from dataclasses import dataclass
import torch

from NEW_py_ver.TOP.TOPProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_prize: torch.Tensor = None
    # shape: (batch, problem)
    max_length: torch.Tensor = None
    # shape: (batch, 1), budget for a single leg
    num_vehicles: torch.Tensor = None
    # shape: (batch, 1) m, how many legs are allowed


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    # selected_count: int = None
    # load: torch.Tensor = None
    # shape: (batch, pomo)
    
    remaining_length: torch.Tensor = None
    # shape: (batch, pomo)
    # gets reset every time you start on a new leg. budget left on CURRENT leg

    legs_remaining: torch.Tensor = None
    # shape: (batch, pomo)
    # additional legs after this one

    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class TOPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_prize = None
        self.saved_max_length = None
        self.saved_num_vehicles = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_prize = None
        # shape: (batch, problem+1)
        self.max_length = None
        # shape: (batch, 1)
        self.num_vehicles = None
        # shape: (batch, 1) 
        
        # Dynamic-1 (path taken so far)
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2 (OP constraint state)
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.remaining_length = None
        # shape: (batch, pomo)
        self.legs_remaining = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()
    
    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_prize = loaded_dict['node_prize']
        self.saved_max_length = loaded_dict['max_length']
        self.saved_num_vehicles = loaded_dict.get('num_vehicles', None)
        self.saved_index = 0
   
    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_prize, max_length, num_vehicles = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_prize = self.saved_node_prize[self.saved_index:self.saved_index+batch_size]
            max_length = self.saved_max_length[self.saved_index:self.saved_index+batch_size]
            num_vehicles = self.saved_num_vehicles[self.saved_index:self.saved_index+batch_size]    
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_prize = node_prize.repeat(8, 1)
                max_length = max_length.repeat(8, 1)
                num_vehicles = num_vehicles.repeat(8, 1)
            else:
                raise NotImplementedError

        self.max_length = max_length
        # shape: (batch, 1)
        self.num_vehicles = num_vehicles
        # shape: (batch, 1)

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_prize = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_prize = torch.cat((depot_prize, node_prize), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_prize = node_prize
        self.reset_state.max_length = max_length
        self.reset_state.num_vehicles = num_vehicles

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        ## if the above throws an error use: # Initialize everyone at the depot (node 0) instead of None to prevent tensor errors
        # self.current_node = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.long, device=self.depot_node_xy.device)

        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        
        self.remaining_length = self.max_length.expand(self.batch_size, self.pomo_size).clone()

        self.legs_remaining = self.num_vehicles.expand(self.batch_size, self.pomo_size).clone()

        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.remaining_length = self.remaining_length
        self.step_state.legs_remaining = self.legs_remaining # claude ver has this substracting the completed legs
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1: extend path and compute distance
        ####################################
        self.selected_count += 1


        prev_node = self.current_node

        # gather coordinates of previous and selected nodes
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)
        prev_xy = all_xy.gather(dim=2, index=prev_node[:, :, None, None].expand(-1, -1, -1, 2)).squeeze(2) # (batch, pomo, 2)
        curr_xy = all_xy.gather(dim=2, index=selected[:, :, None, None].expand(-1, -1, -1, 2)).squeeze(2)  # (batch, pomo, 2)
        step_distance = ((prev_xy - curr_xy) ** 2).sum(dim=2).sqrt() # (batch, pomo)

        self.remaining_length -= step_distance

        ## Update current node and selected node list
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2 Depot Return
        ####################################
        self.at_the_depot = (selected == 0)

        # minus one leg if returned to depot and not the first step (prev_node != 0)
        returned_to_depot = self.at_the_depot & (prev_node != 0)
        self.legs_remaining[returned_to_depot] -= 1

        # reset budget if returned to depot
        next_leg_available = (self.legs_remaining > 0) & self.at_the_depot
        self.remaining_length[next_leg_available] = self.max_length.expand_as(self.remaining_length)[next_leg_available]

        # Masking Logic
        ####################################
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
    
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()

        # Mask the nodes that cannot be visited due to budget constraints
        # Check distances to potential next nodes from current node
        curr_xy_expanded = curr_xy[:, :, None, :].expand(-1, -1, self.problem_size + 1, -1)
        next_node_xy = self.depot_node_xy[:, None, :, :].expand(self.batch_size, self.pomo_size, -1, -1)

        # Distance from current node to all other nodes
        dist_to_next = ((curr_xy_expanded - next_node_xy) ** 2).sum(dim=3).sqrt() # (batch, pomo, problem+1)
        
        # Distance from those next nodes back to the depot (since every leg must finish at depot)
        depot_xy_expanded = self.depot_node_xy[:, [0], None, :].expand(self.batch_size, self.pomo_size, self.problem_size + 1, -1)
        dist_next_to_depot = ((next_node_xy - depot_xy_expanded) ** 2).sum(dim=3).sqrt() # (batch, pomo, problem+1)
        
        # Feasibility check: Can we visit node J and make it back to the depot?
        total_required_budget = dist_to_next + dist_next_to_depot
        insufficient_budget = (self.remaining_length[:, :, None] < total_required_budget)
        self.ninf_mask[insufficient_budget] = float('-inf')

        # Finished if no more legs + all active targets are visited/unreachable
        no_more_legs = (self.legs_remaining == 0)
        no_feasible_moves = (self.ninf_mask[:, :, 1:] == float('-inf')).all(dim=2)
        
        self.finished = no_more_legs | (no_feasible_moves & self.at_the_depot)
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.remaining_length = self.remaining_length
        self.step_state.legs_remaining = self.legs_remaining # calude ver is diff
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_collected_prize()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_collected_prize(self):
        # We collect prizes only from unique visited nodes (excluding depot 0)
        # selected_node_list has shape: (batch, pomo, sequence_length)
        
        # Create a visited mask
        visited_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), device=self.depot_node_xy.device)
        visited_mask.scatter_(dim=2, index=self.selected_node_list, value=1.0)
        visited_mask[:, :, 0] = 0.0 # Ignore depot prizes (which is 0 anyway)
        
        prizes = self.depot_node_prize[:, None, :].expand(-1, self.pomo_size, -1)
        collected_prizes = (visited_mask * prizes).sum(dim=2) # (batch, pomo)
        return collected_prizes


