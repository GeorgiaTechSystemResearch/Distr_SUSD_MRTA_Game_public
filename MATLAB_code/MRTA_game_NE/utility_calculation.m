function u = utility_calculation(agent, P, Y)

% agent: an agent object
% P: the task positions (2 by n_T) where n_T is the number of goal locations
% Y: the task types (n_S by n_T) 

[~, N_tasks] = size(P);
x0 = agent.loc;
Q = agent.Q;
S = Q'*Y;
A = agent.A;

agent_id = agent.id;
assignment = agent.assignments;
task_indices = find(assignment);

if agent.Q(:,agent_id) == [2;1]
    travel_speed = 1.5;
elseif agent.Q(:,agent_id) == [1;2]
    travel_speed = 1.8;
end

%% Method 1: single-robot task utility 
% task_reward-path_cost if it is the only robot to the task, -1*path_cost otherwise

% A_other = agent.A;
% A(agent.id,:) = 0;
% 
% u = 0;
% base_reward = 50;
% [goal_assignment, cost_greedy] = greedy_task_ordering(x0, P(:, task_indices));
% for i = 1 : length(task_indices)
%     task_id = task_indices(i);
%     s_ij = S(agent_id, task_id);
%     task_id = task_indices(i);
%     % path_cost = norm((P(:, task_id)-x0));
%     num_other_robots_selected_the_task = sum(A(:,task_id));
%     if sum(num_other_robots_selected_the_task) == 0
%         task_reward = base_reward*s_ij;
%     else
%         task_reward = 0;
%     end
%     u = u + task_reward;
% end
% u = u - alpha*cost_greedy/travel_speed;

%% Method 2: multi-robot task utility 
effective_specilist_points = 3;
gamma = 1;
y_max = 100;
alpha = 0.3;

A_other = agent.A;
A_other(agent.id,:) = 0;
A_S_other = A_other.*S;

if Q(:,agent_id) == [2;1]
    travel_speed = 1.5;
elseif Q(:,agent_id) == [1;2]
    travel_speed = 1.8;
end
[local_task_ordering, cost_greedy] = greedy_task_ordering(x0, P(:, task_indices));
task_order = task_indices(local_task_ordering);

tot_task_utility = 0;
marginal_task_utilities = zeros(1, length(task_indices));

for i = 1 : length(task_order)
    task_id = task_order(i);
    specilist_points_other = sum(A_S_other(:,task_id));
    s_ij = S(agent_id, task_id);

    % % Marginal contribution (WLU)
    if specilist_points_other < effective_specilist_points
        switch agent.utility_func_type
            case 'linear'
                % linear
                beta = y_max/effective_specilist_points;
                org_task_utility = beta *(specilist_points_other);
                poss_task_utility = beta *(specilist_points_other+s_ij);
                poss_task_utility = min(y_max, poss_task_utility);
                marginal_task_utility = poss_task_utility-org_task_utility;
            case 'convex'
                % convex
                beta = y_max/(exp(effective_specilist_points)-1);
                org_task_utility = beta*(exp(specilist_points_other)-1);
                poss_task_utility= beta*(exp(specilist_points_other+s_ij)-1);
                poss_task_utility = min(y_max, poss_task_utility);
                marginal_task_utility = poss_task_utility-org_task_utility;
            case 'concave'
                % % concave down (submodular)
                beta = y_max/(1-exp(-effective_specilist_points/gamma));
                org_task_utility = beta*(1-exp(-specilist_points_other/gamma)); 
                poss_task_utility = beta*(1-exp(-(specilist_points_other+s_ij)/gamma));
                poss_task_utility = min(y_max, poss_task_utility);
                marginal_task_utility = poss_task_utility-org_task_utility;
        end
    else 
        marginal_task_utility = 0;
    end

    marginal_task_utilities(i) = marginal_task_utility;
end

travel_cost = cost_greedy/travel_speed;
u = sum(marginal_task_utilities) - alpha*travel_cost;

end

