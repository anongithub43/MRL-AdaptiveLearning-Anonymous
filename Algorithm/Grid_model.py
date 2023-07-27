from model import MDP_model
import math
import pandas as pd
import numpy as np 

pd.options.mode.chained_assignment = None # suppressing a warning on chained assignment, 
# reference: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
# for why we need this. 

from testing import get_MDP, next_clusters

# class implementing MDP model that models a space in two feature dimensions as a grid
class Grid_model(MDP_model): 

    def fit(self, df, square_width, maze_size, pfeatures=2, preserve_end_state=False): 
        """
        given dataframe, square width, and maze size arranges the maze into a grid where each 
        square has size square_width. 
        """
        # need to set this because of MDP_model dependencies
        self.pfeatures = pfeatures
        self.cluster_map = {} 

        # let x be square width, i y coordinate, j x coordinate
        # cluster = floor(i/x)*ceil(l/x) + floor(j/x) 

        def get_cluster(feature_0, feature_1, x, l): 
            # x: cell length
            # l: grid length
            i = -feature_1
            j = feature_0
            cluster = math.floor(i/x)*math.ceil(l/x) + math.floor(j/x)

            # if end state is preserved, send the clusters in the 'end box' to the actual end 
            if preserve_end_state: 
                # if cluster in the bottom right square
                reward_square_boundary = l-1
                if i >= reward_square_boundary and j >= reward_square_boundary: 
                    cluster = math.ceil(l/x) + math.floor(l - .00001/x)

            return cluster 

        # set clusters
        if preserve_end_state: 
            # Separate points in/not in the end state
            og_end_idx = maze_size**2 - 1
            end_state_points = df[df['OG_CLUSTER'] == og_end_idx]
            other_points = df[df['OG_CLUSTER'] != og_end_idx]
            
            # Fit all points NOT in end state
            other_points = other_points.copy()
            other_points['CLUSTER'] = other_points.apply(lambda x: get_cluster(x.FEATURE_0, x.FEATURE_1, square_width, maze_size), axis=1)
            # Fit all points in end state
            new_end_idx = np.max(other_points['CLUSTER'].values)+1

            if len(end_state_points) > 0: 
                end_state_points.loc[:, 'CLUSTER'] = new_end_idx # end_state_points['CLUSTER'] = new_end_idx
            
            # Re-combine data
            df = pd.concat([other_points, end_state_points])
            df = df.sort_index()
            df = df.astype({'CLUSTER':'int'})
        else: 
            df['CLUSTER'] = df.apply(lambda x: get_cluster(x.FEATURE_0, x.FEATURE_1, square_width, maze_size), axis=1)

        # removing any clusters that have no datapoints
        unique_clusters = df['CLUSTER'].unique()
        ascending_clusters = sorted(unique_clusters)

        i = 0 
        # # remaps clusters since some grid cells won't have any points and we need clusters to be ascending order
        for cluster in ascending_clusters: 
            df.loc[df['CLUSTER'] == cluster, 'CLUSTER'] = i #df['CLUSTER'][df['CLUSTER'] == cluster] = i
            self.cluster_map[cluster] = i
            i = i + 1
        
        #set next cluster values
        df['NEXT_CLUSTER'] = df['CLUSTER'].shift(-1)
        df.loc[df['ID'] != df['ID'].shift(-1), 'NEXT_CLUSTER'] = 0
        df['NEXT_CLUSTER'] = df['NEXT_CLUSTER'].astype(int)
        df.loc[df['ID'] != df['ID'].shift(-1), 'NEXT_CLUSTER'] = 'None'
        df.loc[df['ACTION']=='None', 'NEXT_CLUSTER'] = 'End'

        self.df_trained = df

        # now that we have set self.df_trained, we can call self.get_MDP()
        self.get_MDP()

        # predictor to predict which cluster a point from the grid lies in. 
        # this is needed for the module
        class Grid_Predictor: 
            def __init__(self, square_width, maze_size, cluster_map): 
                self.square_width = square_width
                self.maze_size = maze_size 
                self.cluster_map = cluster_map
                

            def predict(self, features): 
                # returns the mapping from the defined cluster to what cluster we're actually working with, or sink node
                sink_node = max(self.cluster_map.values()) + 1
                if type(features) == pd.DataFrame: 
                    real_cluster = features.apply(lambda x: self.cluster_map.get(
                                get_cluster(x.FEATURE_0, x.FEATURE_1, square_width, maze_size), sink_node), \
                                axis=1)
                else:
                    theoretical_cluster = get_cluster(features[0][0], features[0][1], self.square_width, self.maze_size)
                    real_cluster = self.cluster_map.get(theoretical_cluster, sink_node)
                return real_cluster
    
        # our 'model' to predict states is just the grid predictor
        self.m = Grid_Predictor(square_width, maze_size, self.cluster_map)
        
    def get_MDP(self): 
        # creates P and R for MDP processes 
        P_df,R_df = get_MDP(self.df_trained)
        
        self.P_df = P_df
        self.R_df = R_df
        
        # store next_clusters dataframe
        self.nc = next_clusters(self.df_trained)


    def aggregate_states(self, gamma, P, R): 

        num_states = P.shape[1] 
        # data structures
        # map each state to the block that it's in 
        state_block_map = {} 
        # map a block to all of its states
        block_state_map = {} 
        # map each block to its reward
        block_reward_map = {} 
        # map each reward to its block
        reward_block_map = {} 
        
        # step 1: cluster by reward 
        # a block just has transition probabilities to other blocks 
        # memorize the transition probabilities between blocks 

        # 1.1 build up which states go to which blocks
        n_blocks = 0
        for i in range(num_states): 
            unhashable_reward = tuple(round(j, 5) for j in R[:, i])
            if unhashable_reward in reward_block_map: 
                block = reward_block_map[unhashable_reward]
                block_state_map[block].add(i) 
                state_block_map[i] = block
            else: 
                state_block_map[i] = n_blocks
                block_state_map[n_blocks] = {i}
                block_reward_map[n_blocks] = unhashable_reward
                reward_block_map[unhashable_reward] = n_blocks
                n_blocks += 1

        # 1.2 build transition matrix 
        D = np.zeros((P.shape[0], num_states, n_blocks))
        for s in range(num_states): 
            # for every state we could transition to, get the block we're transitioning to and add to our probability
            # of transitioning to it 
            for j in range(P.shape[2]): 
                for a in range(P.shape[0]): 
                    D[a, s, state_block_map[j]] += P[a, s, j]

        checked_blocks = np.zeros((D.shape[2], D.shape[2]))

        # step 2: split apart blocks
        incoherent = True
        while incoherent: 
            incoherent = False
            # compare transitions from state b to b_prime
            for b in range(D.shape[2]):
                
                to_print = False
                if block_reward_map[b] == (1.0, 1.0, 1.0, 1.0): 
                    to_print = True

                for b_prime in range(D.shape[2]): 

                    if incoherent:
                        break
                    
                    # making sure we haven't checked this transition yet
                    if checked_blocks[b, b_prime] == 1: 
                        continue
                    
                    # selecting the section of D only for this specific block
                    transformed_state_map = {} 
                    D_block = np.zeros((D.shape[0], len(block_state_map[b])))
                    i = 0
                    for s in block_state_map[b]:
                        D_block[:, i] = D[:, s, b_prime]
                        transformed_state_map[i] = s
                        i = i + 1
                    
                    
                    # now compare all states from the block pair-wise 
                    for i in block_state_map[b]: 

                        if incoherent: 
                            break

                        for j in block_state_map[b]: 

                            if incoherent: 
                                break

                            if j <= i: 
                                continue
                            # split up the block b if not equal
                            action = get_index_inequal_small(D[:, i, b_prime], D[:, j, b_prime], gamma) 

                            if action is not None: 
                                # print(D_block)
                                incoherent = True

                                points = D_block[action, :]

                                free_points_indices = set(i for i in range(points.shape[0]))
                                free_points_values = set(points)

                                # group the points within gamma of each other
                                # TODO: could be sped up to nlogn runtime 
                                blocks = []
                                while len(free_points_indices) > 0: 
                                    block = []
                                    min_point = min(free_points_values)
                                    for i in free_points_indices.copy():
                                        point = points[i] 
                                        if point - min_point < gamma: 
                                            block.append(i)
                                            free_points_indices.remove(i) 
                                            if point in free_points_values: 
                                                free_points_values.remove(point)
                                            
                                    blocks.append(block)

                                block_num = 0

                                # reset D
                                for block in blocks: 
                                    real_states = [transformed_state_map[x] for x in block]
                                    real_block_num = block_num
                                    if real_block_num == 0: 
                                        real_block_num = b
                                        checked_blocks[:, b] = 0 
                                        checked_blocks[b, :] = 0
                                    else: 
                                        # add last blocks to the end so we don't have to overwrite as much 
                                        real_block_num = D.shape[2]
                                        D = np.append(D, np.zeros((D.shape[0], D.shape[1], 1)), axis=2)
                                        checked_blocks = np.append(checked_blocks, np.zeros((1, checked_blocks.shape[0])), axis = 0)
                                        checked_blocks = np.append(checked_blocks, np.zeros((checked_blocks.shape[1] + 1, 1)), axis = 1)

                                    # add corresponding probabilities from P into D for the split blocks
                                    for s in range(P.shape[1]): 
                                        for a in range(P.shape[0]): 
                                            D[a, s, real_block_num] = 0
                                            for block_state in real_states: 
                                                D[a, s, real_block_num] += P[a, s, block_state]

                                    block_num += 1

                                    # reset the maps
                                    block_state_map[real_block_num] = set(real_states)
                                    for state in real_states: 
                                        state_block_map[state] = real_block_num
                                    block_reward_map[real_block_num] = block_reward_map[b]

        # Step 3: reconstruct P and R
        n_blocks = D.shape[2]
        new_P = np.zeros((P.shape[0], n_blocks, n_blocks))
        new_R = np.zeros((R.shape[0], n_blocks))

        # want to order P and R by states, not blocks, so this code gives each
        # block an index into P and R based on its earliest state
        block_index_map = {} 
        current_index = 0 
        for state in range(num_states): 
            block = state_block_map[state]
            
            if block not in block_index_map.keys(): 
                block_index_map[block] = current_index
                current_index += 1

        # actually filling out P here
        for block in range(n_blocks): 
            block_index = block_index_map[block]

            # add transitions from the block to all other blocks
            for state in block_state_map[block]: 
                transition = D[:, state, :]

                for b_prime in range(n_blocks): 
                    new_P[:, block_index, block_index_map[b_prime]] += transition[:, b_prime] / len(block_state_map[block])
            
            # add rewards of the block 
            block_reward = block_reward_map[block]
            for i in range(len(block_reward)): 
                new_R[i, block_index] = block_reward[i]
        
    
        for cluster, state in self.cluster_map.items(): 
            self.cluster_map[cluster] = block_index_map[state_block_map[state]]

        return new_P, new_R


    def old_aggregate_states(self, gamma, P, R):
        """
        performs state aggregation
        P, R -> Pagg, Ragg
        where if 
        |P(s1, *) - Pa(s2, *)| < gamma we aggregate states s1 and s2.
        P(s1, *) is the vector of probabilities of s1 to transition to all other states. 

        TODO: add kmeans clustering algorithm 

        Can a single cell merge multiple times in one iteration? 
        if states i and j are merging, and j and k are merging, should we merge i, j, and k into one big state, or just do i and j? 
        """

        # if we don't find an aggregation, finish
        found = True

        # keep matrix D[i,j] = |P_i - P_j| 
        # keep track of all states that are still relevant and filter for them at the end 
        # keep in memory all merges or try dynamically 
        # list/ structure of states that contains what we merged 

        # P has dimensions a x s x s (4 x 2500 x 2500 = 25,000,000 for 5x5)
        # s number of clusters
        
        # D = s x s 
        # matrix that is 0 if the spot in D is filled and 1 otherwise

        s = P.shape[1]

        D_tried = np.zeros((s, s))
        while found: 
            merging_states = set() 
            states_to_merge = []
            states_to_remove = [] 
            found = False

            s = P.shape[1]

            # using s-2 so we don't aggregate any of the sinks 
            for i in range(s): 
                s_1 = P[:, i, :]
                # TODO: speed up by allowing multiple merge states in one iteration
                if i in merging_states: 
                    continue 
                for j in range(s): 
                    if i == j or D_tried[i, j] == 1 or j in merging_states: 
                        continue
                    s_2 = P[:, j, :]

                    # taking the norm of the difference between the matrices
                    diff = np.linalg.norm(s_1 - s_2)

                    D_tried[i, j] = True

                    # if we find a match, remember what we need to do to merge the states
                    if diff < gamma: 
                        found = True
                        merging_states.add(i)
                        merging_states.add(j)
                        states_to_merge.append((i, j))
                        states_to_remove.append(j)

            # now, iterate over to_merge and combine the states
            for states in states_to_merge: 
                i, j = states 
                # print(i)
                # print(P[:, i, :])
                # print(j)
                # print(P[:, j, :])
                # print('-----------')
                # print(np.linalg.norm(P[:, i, :] - P[:, j, :]))
                from_states = 0.5*(P[:, i, :] + P[:, j, :])

                P[:, i, :] = from_states.copy()

                to_states = P[:, :, i] + P[:, :, j]
                # print(to_states)
                # print(from_states)
                P[:, :, i] = to_states.copy()
                
                R[:, i] = 0.5*(R[:, i] + R[:, j])
                # reset the values of i that we have tried since the state is new 
                D_tried[i, :] = np.zeros(s)
                D_tried[:, i] = np.zeros(s)

                # whatever cluster mapped to j now maps to i 
                j_clusters = [k for k,v in self.cluster_map.items() if v == j]
                for clus in j_clusters: 
                    self.cluster_map[clus] = i

            # remove the states to remove
            removed_states = set()
            for state in states_to_remove: 
                
                # take into account what we've already removed 
                effective_state = state
                for removed in removed_states: 
                    if removed < state: 
                        effective_state = effective_state - 1
                
                #print(effective_state)
                #print(P.shape)

                # set all clusters back one when merging cluster maps
                new_map = {}
                for k,v in self.cluster_map.items(): 
                    if v < j: 
                        new_map[k] = v
                    else: 
                        new_map[k] = v - 1
                self.cluster_map = new_map
                
                P = np.delete(P, effective_state, 1)
                P = np.delete(P, effective_state, 2)
                R = np.delete(R, effective_state, 1)
                D_tried = np.delete(D_tried, effective_state, 0)
                D_tried = np.delete(D_tried, effective_state, 1)
                removed_states.add(state)
                
                # total number of states decreases by 1 
                s = s - 1 

        return P, R

    def solve_MDP(self,
                  alpha = 0.2, # statistical alpha threshold
                  beta = 0.6, # statistical beta threshold
                  min_action_obs = -1, # int: least number of actions that must be seen
                  min_action_purity = 0.3, # float: percentage purity above which is acceptable
                  prob='max', # str: 'max', or 'min' for maximization or minimization problem
                  gamma=0.9, # discount factor
                  epsilon=10**(-10),
                  p=False, 
                  agg_gamma=1,
                  algorithm='stochastic',
                  aggregate=False):
        
        if algorithm == 'stochastic': 
            self.P, self.R = get_Stochastic_Empirical_MDP(self.df_trained)
        else: 
            self.create_PR(alpha, beta, min_action_obs, min_action_purity, prob)
        n_states = self.P.shape[1]
        if aggregate: 
            self.P, self.R = self.aggregate_states(agg_gamma, self.P, self.R)
            #print('new number of states: ', self.P.shape[1])
            #print('old number of states: ', n_states)
        return self.solve_helper(gamma, epsilon, p, prob, threshold=0) # threshold an unused parameter in the function

def get_index_inequal_small(a, b, gamma): 
    # returns the first index at which a and b are different by more than gamma
    for i in range(a.shape[0]): 
        if abs(a[i] - b[i]) > gamma:
            return i
    return None

def get_df_attributes(trained_df): 
    #a = trained_df['ACTION'].nunique() - 1
    #s = trained_df['CLUSTER'].nunique()
    actions = trained_df['ACTION'].unique()
    actions = np.delete(actions, np.where(actions == 'None'))
    clusters = trained_df['CLUSTER'].unique()
    a = len(actions)
    s = len(clusters)    
    return a, s, actions, clusters

def construct_R(trained_df, t_max=25): 
    '''
    the reward of each cluster is the average reward of all points in the cluster 
    '''
    a, s, actions, clusters = get_df_attributes(trained_df)
    
    R = np.zeros((s,))
    for cluster in clusters: 
        in_cluster = trained_df[trained_df['CLUSTER'] == cluster]
        reward = np.mean(in_cluster['RISK'])
        R[cluster] = reward
    
    # Add Sink and Punishment node  
    r_max = np.max(R)
    punishment = -t_max*r_max*2
    R = np.append(R, [0, punishment])
    # duplicate for each action
    R = np.repeat(np.array([R]), a, axis=0)
    return R 
        
def construct_P(trained_df): 
    '''
    P[a, s, s'] = (# pts that were in s and took a { AND got to s }')/(# pts that were in s and took a)
    '''
    num_actions, num_states, actions, clusters = get_df_attributes(trained_df)
    
    end = max(clusters)
    sink = end+1 
    punish = end+2
    
    P = np.zeros((num_actions, num_states+2, num_states+2)) 
                                # punishment state always goes to sink state, sink state always stays there
    
    # iterate over all possible combinations of (a, s, s')
    for a in actions: # TODO: switch back to i, a for accurate code
        i = int(a)
        # map punish node to sink node no matter what
        P[i, punish, sink] = 1
        # map end state to sink node no matter what
        P[i, end, sink] = 1
        # sink state always stays no matter what
        P[i, sink, sink] = 1
            
        for j, s in enumerate(clusters):
            if s != end: 
                # get rows where CLUSTER = s and ACTION = a
                subset = trained_df[(trained_df["CLUSTER"] == s) & (trained_df["ACTION"] == a) & (trained_df["NEXT_CLUSTER"] != 'None')]
                total_count = subset.shape[0]

                if total_count == 0:  # if no data
                    P[i, s, punish] = 1

                else:
                    # count the number of rows with NEXT_CLUSTER = s'
                    for k, s_prime in enumerate(clusters):
                        correct = subset[subset["NEXT_CLUSTER"] == s_prime]
                        correct_count = correct.shape[0]
                        '''if i==1 and s==7:
                            print(correct)
                            print(i, s, s_prime)
                            print(correct_count / total_count)'''

                        # calculate P[a, s, s']
                        P[i, s, s_prime] = correct_count / total_count #P[i, j, k] = correct_count / total_count # TODO: double check mapping

    return P 
    
def get_Stochastic_Empirical_MDP(trained_df): 
    '''
    gridded_df: df of maze trial run datapoints partitioned/clustered into a grid
    Returns: P, R representing MDP 
    '''
    R = construct_R(trained_df)
    P = construct_P(trained_df)
    return P,R