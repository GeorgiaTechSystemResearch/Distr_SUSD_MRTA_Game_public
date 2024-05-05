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

reward_selecting_an_empty_task = 50;

if agent.Q(:,agent_id) == [2;1] % H-AUV
    travel_speed = 1.0;
    power_travel = 115/1000;
elseif agent.Q(:,agent_id) == [1;2] % L-AUV
    travel_speed = 1.3;
    power_travel = 95/1000;
end

power_cost_LAUV_scan = 100/1000;
power_cost_LAUV = 95/1000;
dur_LAUV_scan = 30;
dur_LAUV_inspect = 60;

power_cost_HAUV = 115/1000;
dur_HAUV_inspect= 30;
dur_HAUV_scan = 60;

alpha = 1;

A_other = agent.A;
A_other(agent.id,:) = 0;

[local_task_ordering, cost_greedy] = greedy_task_ordering(x0, P(:, task_indices));
task_order = task_indices(local_task_ordering);

marginal_task_utilities = zeros(1, length(task_indices));
for i = 1 : length(task_order)
    task_id = task_order(i);
    
    y_j = Y(:,task_id);
    % q_j = Q* A(:,task_id);
    % q_j_other = Q* A_other(:,task_id);
    q_j  = Q.*A(:,task_id)';
    q_j_other = Q.* A_other(:,task_id)';
    N_inspect = y_j(1);
    N_survey = y_j(2);

    e_A = power_cost_HAUV*dur_HAUV_inspect*N_inspect + power_cost_LAUV_scan*dur_LAUV_scan*N_survey; % 1 H-AUV + 1 L-AUV
    e_B = power_cost_LAUV*dur_LAUV_inspect*N_inspect + power_cost_LAUV_scan*dur_LAUV_scan*N_survey; % L-AUV
    e_C = power_cost_HAUV*(dur_HAUV_inspect*N_inspect + dur_HAUV_scan*N_survey); % H-AUV
    y_max = max(e_C, e_B)+abs(e_C-e_B);
    num_HAUV = sum(q_j_other(1,:) == 2);
    num_LAUV = sum(q_j_other(1,:) == 1);
    if num_HAUV > 0 && num_LAUV >0 % 1 H-AUV + 1 L-AUV
        org_task_utility = y_max-e_A;
    elseif num_LAUV > 0 %  L-AUV
        org_task_utility = y_max-e_B;
    elseif num_HAUV > 0 %  H-AUV
        org_task_utility = y_max-e_C;
    else
        org_task_utility = 0;
    end
    
    num_HAUV = sum(q_j(1,:) == 2);
    num_LAUV = sum(q_j(1,:) == 1);
    if num_HAUV > 0 && num_LAUV >0 % 1 H-AUV + 1 L-AUV
        poss_task_utility = y_max- e_A;
    elseif num_LAUV > 0 %  L-AUV
        poss_task_utility = y_max- e_B;
    elseif num_HAUV > 0 %  H-AUV
        poss_task_utility = y_max- e_C;
    else
        poss_task_utility = 0;
        error('Error: should not be here')
    end

    marginal_task_utility = poss_task_utility-org_task_utility;
    marginal_task_utilities(i) = marginal_task_utility;
end

travel_cost = power_travel*cost_greedy/travel_speed;
% Penalty for selecting empty tasks
% num_empty_tasks = N_tasks - nnz(sum(A,1));
% penalty = 0 * num_empty_tasks;

% Bonus on selecting a emptytask
N_empty_task_selected = length(intersect(task_order, find(sum(A_other,1) ==0)));
reward = reward_selecting_an_empty_task * N_empty_task_selected;
u = sum(marginal_task_utilities) - alpha*travel_cost + reward;

end

