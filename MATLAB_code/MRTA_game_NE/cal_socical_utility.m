function socical_utility = cal_socical_utility(agents, P, Y)
    [~, N_agents] = size(agents);
    [~, N_tasks] = size(P);
    
    effective_specilist_points = 3;
    gamma = 1;
    y_max = 100;
    alpha = 0.3;
    
    socical_utility = 0;
    total_travel_cost = 0;
    for a_i = 1:N_agents
        agent_i = agents(a_i);
        x0 = agent_i.loc;
        agent_id = agent_i.id;
        if agent_i.Q(:,agent_id) == [2;1]
            travel_speed = 1.5;
        elseif agent_i.Q(:,agent_id) == [1;2]
            travel_speed = 1.8;
        end

        task_indices = agent_i.selected_tasks;
        [local_task_ordering, cost_greedy] = greedy_task_ordering(x0, P(:, task_indices));
        task_ordering = task_indices(local_task_ordering);
        
        P1 = P(:,task_ordering);
        P0 = [x0, P1(:,1:end-1)];
        dist_vec = sqrt(sum((P1-P0).^2, 1));
        travel_cost = sum(dist_vec)/travel_speed;
        total_travel_cost = total_travel_cost + travel_cost;
    end
    
    A = agents(1).A;
    Q = agents(1).Q;
    S = Q'*Y;
    A_S = A.*S;
    task_utilities_vec = zeros(1, N_tasks);
    func_type = agents(1).utility_func_type;
    for j = 1:N_tasks
        specilist_points = sum(A_S(:,j));
        switch func_type
            case 'linear'
                % Linear
                beta = y_max/effective_specilist_points;
                task_utility = beta * specilist_points;
            case 'convex'
                % Convex 
                beta = y_max/(exp(effective_specilist_points)-1);
                task_utility = beta*(exp(specilist_points)-1);
            case 'concave'
                % Concave
                beta = y_max/(1-exp(-effective_specilist_points/gamma));
                task_utility = beta*(1-exp(-specilist_points/gamma)); 
        end
        task_utility = min(y_max, task_utility);
        task_utilities_vec(1, j) = task_utility;
     end
    
    socical_utility = sum(task_utilities_vec) -  alpha*total_travel_cost;

end

