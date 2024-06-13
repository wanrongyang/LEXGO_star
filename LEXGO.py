
import networkx as nx
import numpy as np
import time


class LEXGO:
    def __init__(self, graph, start_node, destination, lex_preference):
        self.graph = graph
        self.start = start_node
        self.destination = destination
        self.OPEN = [{'s': [(float(0), float(0)), (10, 8, 4), (0, 0, 0)]}]
        self.COSTS = []
        # Here is only 2 priority levels, so there are 2 values in d_b
        # If there are 3 priority levels, then there are 3 values in d_b
        self.d_b = tuple(float('inf') for _ in range(len(lex_preference)))
        self.G_op = {str(self.start): [(0, 0, 0)], 'n1': [], 'n2': [], 'n3': [], 't': []}
        self.G_cl = {}
        self.lgp = lex_preference
        # Heuristic function, this is given in the original paper
        # Different examples may need to be calculated by estimating the lower bound function
        self.h_n = {'s': (10, 8, 4), 'n1': (8, 6, 5), 'n2': (7, 5, 2), 'n3': (5, 4, 2), 't': (0, 0, 0)}
        self.weights_format = list(self.graph.edges(data='weight'))[0][2]
        self.weights_length = len(self.weights_format)
        self.SG = nx.DiGraph()
        self.SG.add_node(self.start)
        self.current_n = self.start
        self.g_n = self.G_op[self.start][0]
        self.d_n = self.simple_deviation_vector(x=self.g_n)

    # < Dominance
    @staticmethod
    def pareto_optimal_preference(x, y):
        """
        Check if vector y Pareto-optimal preference dominates vector y_prime.

        Args:
            x (tuple or numpy.ndarray or list): The first vector.
            y (tuple or numpy.ndarray or list): The second vector.

        Returns:
            bool: True if x Pareto-optimal preference dominates y, False otherwise.
        """
        # Check if x and y have the same length
        x = list(x)
        y = list(y)

        if len(x) != len(y):
            raise ValueError(
                "Vectors x and y must have the same length".format(len(x), len(y)))
        else:
            pass

        # Check if all elements in x are less than or equal to the corresponding elements in y
        if all(a <= b for a, b in zip(x, y)):
            # Check if at least one element in x is strictly less than the corresponding element in y
            if any(a < b for a, b in zip(x, y)):
                return True

        # if any(a < b for a, b in zip(x, y)):
        return False

    @staticmethod
    # <L: Lexicographic order
    def lexicographic_order(x, y):
        x = list(x)
        y = list(y)

        if len(x) != len(y):
            raise ValueError(
                "Vectors x and y must have the same length".format(len(x), len(y)))
        else:
            pass

        for a, b in zip(x, y):
            if a == b:
                pass
            elif a < b:
                return True
            elif a > b:
                return False
            else:
                # the 2 vectors are equal
                return False

    def simple_path_cost(self, current_node, successor):

        all_simple_paths = list(nx.all_simple_paths(self.graph, source=current_node, target=successor))
        print(all_simple_paths)

        for path in all_simple_paths:
            total_costs = np.zeros(self.weights_length)
            for i in range(len(path)):
                if i == len(path) - 1:
                    break
                else:
                    edge_cost = self.graph[path[i]][path[i + 1]]['weight']
                    total_costs += np.array(list(edge_cost))
            # Add the cumulative cost vector as the last element of each path
            path.append(tuple(total_costs))

        return all_simple_paths

    def simple_deviation_vector(self, x):
        # Only used to bias vector based on x vector and preference target value
        x = list(x)
        lex_prefer_goals = self.lgp
        level_num = len(lex_prefer_goals)
        # Store the cumulative weighted deviation value of each level (importance)
        dev_vector = []
        for lexico_level in range(level_num):
            lex_goals = lex_prefer_goals[str(lexico_level)]
            total_deviation_for_each_level = 0
            for each_goal in lex_goals:
                # 每一个goal是(1, 10, 0.5)是这样的形式，goal[0]是属性编号，goal[1]是目标值，goal[2]是权重
                dev = max(0, x[each_goal[0]] - each_goal[1]) * each_goal[2]
                total_deviation_for_each_level = total_deviation_for_each_level + dev
            dev_vector.append(float(total_deviation_for_each_level))

        return dev_vector

    # <G: lexicographic goal preference
    def lexicographic_goal_preference(self, x, y):

        # Check if x and y have the same length
        x = list(x)
        y = list(y)

        if len(x) != len(y):
            raise ValueError(
                "Vectors x and y must have the same length".format(len(x), len(y)))
        else:
            pass

        # 计算他们两个的偏差向量
        x_deviation = self.simple_deviation_vector(x)
        y_deviation = self.simple_deviation_vector(y)
        if self.lexicographic_order(x_deviation, y_deviation):
            return True
        # elif x_deviation == y_deviation and self.pareto_optimal_preference(x, y):
        elif x_deviation == y_deviation and self.lexicographic_order(x, y):
            return True
        else:
            return False

        # condition_1 = self.lexicographic_order(x_deviation, y_deviation)
        # condition_2 = x_deviation == y_deviation and self.pareto_optimal_preference(x, y)
        #
        # if condition_1 or condition_2:
        #     return True
        # else:
        #     return False

    def cross_slack_vector(self, x, y):

        x = list(x)
        y = list(y)

        cross_slacks = []
        for lexico_level in range(len(self.lgp)):
            lex_goals = self.lgp[str(lexico_level)]
            cross_slack_for_each_level = 0
            for each_goal in lex_goals:
                x_slack = max(0, each_goal[1] - x[each_goal[0]])
                y_slack = max(0, each_goal[1] - y[each_goal[0]])
                cross_slack_on_attribute = max(0, y_slack - x_slack) * each_goal[2]
                cross_slack_for_each_level = cross_slack_for_each_level + cross_slack_on_attribute
            cross_slacks.append(float(cross_slack_for_each_level))
        return cross_slacks

    # <P
    def pruning_preference(self, x, y):
        # Check if x allows pruning y
        x = list(x)
        y = list(y)

        if len(x) != len(y):
            raise ValueError(
                "Vectors x and y must have the same length".format(len(x), len(y)))
        else:
            pass

        cross_slacks_on_each_level = self.cross_slack_vector(x, y)
        # Initialize all conditions to False
        condition_1 = False
        condition_2 = False

        x_deviation = self.simple_deviation_vector(x)
        y_deviation = self.simple_deviation_vector(y)
        # Check if the first condition is met
        index = 0
        for j in range(len(self.lgp)):
            if x_deviation[j] < y_deviation[j]:
                condition_1 = True
                if cross_slacks_on_each_level[j] < y_deviation[j] - x_deviation[j]:
                    condition_2 = True
                    index = j
                    break
                else:
                    condition_2 = False
                    pass
            else:
                condition_1 = False
                pass

        def check_condition_3(i):
            if x_deviation[i] == y_deviation[i]:
                if cross_slacks_on_each_level[i] == 0:
                    condition_3 = True
                    return condition_3
                else:
                    condition_3 = False
                    return condition_3
            else:
                condition_3 = False
                return condition_3

        all_satisfy = all(check_condition_3(i) for i in range(index))

        if condition_1 and condition_2 and all_satisfy:
            # print(f"{x} allows to prune {y}")
            return True
        else:
            # print(f"{x} do not allows to prune {y}")
            return False

    def evaluation_vector(self, node):
        # It is to calculate h_n when g_n is known. It is assumed here that it is known
        g_n = self.G_op[node]
        # Minimum path estimate value (determined by other references)
        h_n = self.h_n[node]
        f_n = np.array(list(g_n)) + np.array(list(h_n))
        return tuple(f_n)

    # The deviation vector is based on f_n, not just the cumulative cost vector g_n
    # This can better contain global deviation information.
    def deviation_vector(self, node):
        level_num = len(self.lgp)
        # 每一个node都有一个从起点出发到这个节点的成本，等于每条路径上的对应的属性成本的和
        # g_n represents the cost from the starting point to a certain node, which is a tuple
        f_n = self.evaluation_vector(node)
        devi_vector = []
        for i in range(level_num):
            goals = self.lgp[str(i)]
            total_deviation_for_each_level = 0
            for goal in goals:
                # Each goal is in the form of (0, 10, 0.5)
                # goal[0] is the attribute number, goal[1] is the target value, goal[2] is the weight
                # Attribute numbers start from 0, 0 represents the first attribute
                dev = max(0, f_n[goal[0]] - goal[1]) * goal[2]
                total_deviation_for_each_level = total_deviation_for_each_level + dev
            devi_vector.append(float(total_deviation_for_each_level))
        return tuple(devi_vector)

    def pareto_deviation_filtering(self, label):

        key, value = list(label.items())[0]
        d_n = value[0]
        f_n = value[1]

        # pareto_filtering
        if all(self.pareto_optimal_preference(x=f_n, y=c_start) for c_start in self.COSTS):
            # deviation based filtering
            if self.lexicographic_order(x=self.d_b, y=d_n):
                # Current best achievement vector is better than the new d_n, so it can be pruned
                return True
            else:
                # It means that the new d_n may be better than the current d_b, so no need to prune.
                return False
        else:
            # Means that there is a c_start that is better than f_n, then f_n can be pruned.
            return True

    def para_filtering(self, label):

        if not self.COSTS:

            return False

        else:

            key, value = list(label.items())[0]
            f_n = value[1]

            if any(self.pareto_optimal_preference(x=c_start, y=f_n) for c_start in self.COSTS):
                print(f"{label} should be para_to filtered: ")
                return True
            else:
                return False

    def deviation_filtering(self, label):

        if self.d_b == (float('inf'), float('inf')):

            return False

        else:
            key, value = list(label.items())[0]
            d_n = value[0]

            if self.lexicographic_order(x=self.d_b, y=d_n):

                print(f"{label} should be deviation filtered: ")
                return True
            else:
                return False

    def union_gop_gcl(self, current_node):

        # Check if G_cl is empty first
        if not self.G_cl:
            # G_cl is empty
            union = set(self.G_op[current_node])
        else:
            # G_cl is not empty

            # Then check whether the current node exists in G_cl
            if current_node in self.G_cl:
                union = set(self.G_op[current_node]).union(set(self.G_cl[current_node]))
            else:
                union = set(self.G_op[current_node])

        return union

    def pareto_pruning(self, g_m, current_node):

        if any(self.pareto_optimal_preference(x=g, y=g_m) for g in self.union_gop_gcl(current_node)):

            print(f"Pareto Pruned: {g_m}")
            return True
        else:
            return False

    def deviation_based_pruning(self, f_m, current_node):

        def g_plus_h_m(g):

            return np.array(list(g)) + np.array(self.h_n[current_node])

        if any(self.pruning_preference(x=g_plus_h_m(g), y=f_m) for g in self.union_gop_gcl(current_node)):

            print(f"Deviation based Pruned: {f_m}")
            return True
        else:
            return False

    def pruning(self, g_m, f_m, current_node):

        # If one of the two agrees to cut it, it means that the new_label will be cut.
        p_pruning = self.pareto_pruning(g_m=g_m, current_node=current_node)
        d_pruning = self.deviation_based_pruning(f_m=f_m, current_node=current_node)

        if p_pruning or d_pruning:
            return True
        else:
            return False

    def eliminate_label_from_open(self, current_node):

        # OPEN = [{'s': [(float(0), float(0)), (10, 8, 4), (0, 0, 0)]}]
        for label in self.OPEN:
            if current_node == list(label.keys())[0]:
                self.OPEN.remove(label)
                break
            else:
                continue

    def gop_to_gcl(self, label):

        current_node = list(label.keys())[0]
        g_n = list(label.values())[0][2]

        for cost_vectors in self.G_op[current_node]:
            if g_n == cost_vectors:
                self.G_op[current_node].remove(g_n)
                # Make sure key current_node exists in the dictionary and its value is a list
                self.G_cl.setdefault(current_node, [])
                self.G_cl[current_node].append(g_n)
                break
            else:
                continue

    def path_selection(self):

        def lgp_comparison_fn_labels(label_1, label_2):

            key_1, value_1 = list(label_1.items())[0]
            key_2, value_2 = list(label_2.items())[0]

            if self.lexicographic_goal_preference(x=value_1[1], y=value_2[1]):
                return True
            else:
                return False

        # cloned_open = self.OPEN.copy()
        sub_open = self.OPEN.copy()

        # Check if OPEN is empty
        if not self.OPEN:
            print("OPEN is empty. Algorithm terminated.")
            return None

        else:
            for label in self.OPEN:
                sub_open.remove(label)

                if not sub_open:
                    selected_label = label
                    # Only one label left in OPEN
                    self.OPEN.remove(selected_label)
                    chosen_node = list(selected_label.keys())[0]
                    self.current_n = chosen_node
                    self.g_n = list(selected_label.values())[0][2]
                    print(f"Selected label: {selected_label}")
                    self.d_n = list(selected_label.values())[0][0]
                    return selected_label

                elif not any(lgp_comparison_fn_labels(other_label, label) for other_label in sub_open):
                    selected_label = label
                    self.OPEN.remove(selected_label)
                    self.gop_to_gcl(label=selected_label)
                    f_n = list(selected_label.values())[0][1]

                    if all(self.pareto_optimal_preference(x=f_n, y=c_start) for c_start in self.COSTS):
                        # f_n is a good enough solution
                        chosen_node = list(selected_label.keys())[0]
                        self.current_n = chosen_node
                        self.g_n = list(selected_label.values())[0][2]
                        print(f"Selected label: {selected_label}")
                        self.d_n = list(selected_label.values())[0][0]
                        return selected_label

                    else:
                        # f_n is not a good enough solution
                        continue
                else:
                    sub_open = self.OPEN.copy()
                    # Check if the label is the last one in OPEN
                    if label == self.OPEN[-1]:
                        print("No label is selected. Algorithm terminated.")
                        print(f"Open is: {self.OPEN}, but no label is selected.")
                        return False
                    else:
                        continue

    def solution_recording(self):

        g_n = self.g_n

        # g_n = self.g_n
        self.COSTS.append(g_n)

        self.d_b = self.d_n

    def path_expansion(self, current_node):
        # standing on the current node
        # And then expand to all reachable nodes
        successors = list(self.graph.successors(current_node))
        for successor in successors:
            # extract the edge costs (weights) between the current node and the successor

            edge_costs = self.graph[current_node][successor]['weight']

            # If there is only one edge between these two points
            # Then this edge_costs is a tuple
            # If there are multiple edges between these two vertexes
            # Then this edge_costs is a list of tuples

            edge_costs = [edge_costs]

            h_m = list(self.h_n[successor])
            h_m = np.array(h_m)

            for each_cost in edge_costs:

                cost_n_m = each_cost
                cost_n_m = np.array(list(cost_n_m))
                g_n = np.array(list(self.g_n))

                g_m = g_n + cost_n_m
                f_m = g_m + h_m

                # Changes the format of the label into list
                f_m = f_m.tolist()
                g_m = g_m.tolist()

                # print(f"Current f_m: {f_m}")
                d_m = self.simple_deviation_vector(x=f_m)

                # Changes the format of the label into tuple
                g_m = tuple(g_m)
                d_m = tuple(d_m)
                f_m = tuple(f_m)

                new_label = {str(successor): [d_m, f_m, g_m]}

                # Pareto-deviation filtering
                if self.para_filtering(label=new_label) or self.deviation_filtering(label=new_label):
                    # Be pruned and not add to OPEN
                    # Continue to the next successor
                    print(f"Filtered label: {new_label}")
                    continue
                else:

                    if self.SG.has_node(successor) is False:
                        # if new_label not in self.OPEN:
                        # Make sure there is no such node in SG
                        self.OPEN.append(new_label)
                        # Add the new label's total costs to G_op(s)
                        self.G_op[str(successor)].append(g_m)
                        # Just recording the path
                        self.SG.add_node(successor)
                    elif g_m in self.union_gop_gcl(successor):
                        print(f"label already in, then just pass: {new_label}")
                        # Just recording the path
                        self.SG.add_node(successor)
                    elif self.pruning(g_m=g_m, f_m=f_m, current_node=successor) is False:
                        # i. Eliminate vector g_m_prime from G_op(s)
                        for g_m_p in self.G_op[successor]:
                            g_m_p_h_m = np.array(list(g_m_p)) + np.array(list(self.h_n[successor]))

                            if self.pareto_optimal_preference(g_m, g_m_p):
                                # If g_m dominates g_m_p, then prune g_m_p
                                self.G_op[successor].remove(g_m_p)
                                self.eliminate_label_from_open(current_node=successor)
                                print(f"eliminated g_m_prime: {g_m_p}")
                            elif self.pruning_preference(x=f_m, y=g_m_p_h_m):
                                # If f_m dominates g_m_p_h_m, then prune g_m_p
                                self.G_op[successor].remove(g_m_p)
                                # Delete the corresponding Label from OPEN
                                self.eliminate_label_from_open(current_node=successor)
                                print(f"eliminated g_m_prime(pruning_preference): {g_m_p}")

                        # ii. Add the new label to OPEN
                        self.OPEN.append(new_label)
                        self.G_op[str(successor)].append(g_m)
                    else:
                        continue

    def running(self):

        running_start = time.time()
        count = 0

        while True:

            count += 1
            print(f"#####Current iteration####: {count}")

            # Termination condition
            if not self.OPEN or self.lexicographic_order(x=self.d_b, y=self.d_n):
                print("OPEN is empty. Algorithm terminated.")
                print(f"Optimal cost vectors: {self.COSTS}")

                running_end = time.time()

                print(f"Running time: {running_end - running_start}")

                return self.COSTS
            else:
                print(f"## Current OPEN ##: {self.OPEN}")
                # 2. Path selection
                self.path_selection()
                # 5. Path expansion
                self.path_expansion(current_node=self.current_n)
                # Destination check
                if self.current_n == self.destination:
                    print(f"Found goal node: {self.destination}")
                    # 4. Solution recording
                    self.solution_recording()
                    print(f"Solution recorded: {self.COSTS}")
                else:
                    continue


# main function
def main():
    # define a directed graph
    graph = nx.DiGraph()

    # Given nodes and edges (example in the original paper)
    edges = [
             ('s', 'n1', {'weight': (2, 2, 2)}),
             ('s', 'n3', {'weight': (7, 6, 2)}),
             ('s', 'n2', {'weight': (3, 3, 6)}),
             ('n1', 'n3', {'weight': (3, 3, 3)}),
             ('n1', 't', {'weight': (8, 6, 8)}),
             ('n3', 't', {'weight': (5, 4, 2)}),
             ('n2', 'n3', {'weight': (2, 2, 2)}),
             ('n2', 't', {'weight': (9, 5, 2)})
            ]

    graph.add_edges_from(edges)

    # Attribute number starts from 0
    # The string '0' in lgp represents the first priority level, and the string '1' represents the second priority.
    # Define the start node, goal node, and lexicographic goal preference (lgp)
    start = 's'
    goal = 't'
    lgp = {'0': [(0, 10, 0.5), (1, 10, 0.5)], '1': [(2, 10, 1)]}

    # Initialize the LEXGO* algorithm
    lexgo = LEXGO(graph=graph, start_node=start, destination=goal, lex_preference=lgp)

    # Run the LEXGO* algorithm

    lexgo.running()


print("Running LEXGO algorithm...")

if __name__ == '__main__':
    main()
