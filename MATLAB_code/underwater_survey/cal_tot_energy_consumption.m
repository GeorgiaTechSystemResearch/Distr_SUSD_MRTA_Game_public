function tot_energy_consumption = cal_tot_energy_consumption(agents, P, Y)
    [~, N_agents] = size(agents);
    [~, N_tasks] = size(P);
    
    y_min = 60;
    y_max = agents(1).t_max;
    gamma = 1;


    power_cost_LAUV_scan = 100/1000;
    power_cost_LAUV = 95/1000;
    dur_LAUV_scan = 30;
    dur_LAUV_inspect = 60;
    
    power_cost_HAUV = 115/1000;
    dur_HAUV_inspect= 30;
    dur_HAUV_scan = 60;

    
    total_travel_cost = 0;
    for a_i = 1:N_agents
        agent_i = agents(a_i);
        x0 = agent_i.loc;
        agent_id = agent_i.id;
        if agent_i.Q(:,agent_id) == [2;1]
            travel_speed = 1.0;
            power_travel = 115/1000;
        elseif agent_i.Q(:,agent_id) == [1;2]
            travel_speed = 1.3;
            power_travel = 95/1000;
        end

        task_indices = agent_i.selected_tasks;
        [local_task_ordering, cost_greedy] = greedy_task_ordering(x0, P(:, task_indices));
        task_ordering = task_indices(local_task_ordering);
        agents(a_i).path = task_indices(local_task_ordering);
        
        P1 = P(:,task_ordering);
        P0 = [x0, P1(:,1:end-1)];
        dist_vec = sqrt(sum((P1-P0).^2, 1));
        travel_cost = power_travel*sum(dist_vec)/travel_speed;
        total_travel_cost = total_travel_cost + travel_cost;
    end
    
    A = agents(1).A;
    Q = agents(1).Q;
    for a_i = 1:N_agents
        A(a_i,:) = agents(a_i).A(a_i,:);
    end 
    task_cost_vec = zeros(1, N_tasks);
    optimal_task_energy_vec = zeros(1, N_tasks);
    for j = 1:N_tasks
        task_id = j;
        y_j = Y(:,task_id);
        N_inspect = y_j(1);
        N_survey = y_j(2);
        q_j  = Q.*A(:,task_id)';
        num_HAUV = sum(q_j(1,:) == 2);
        num_LAUV = sum(q_j(1,:) == 1);

        % if all(y_j == [0, 1]') % scanning
        %     if num_LAUV > 0 %  L-AUV
        %         task_cost = power_cost_LAUV_scan*dur_LAUV_scan;
        %     elseif num_HAUV > 0 %  H-AUV
        %         task_cost = power_cost_HAUV*dur_HAUV_scan;
        %     end
        % 
        % elseif all(y_j == [1, 0]') %  inspection
        %     if num_LAUV > 0 %  L-AUV
        %         task_cost = power_cost_LAUV*dur_LAUV_inspect;
        %     elseif num_HAUV > 0 %  H-AUV
        %         task_cost = power_cost_HAUV*dur_HAUV_inspect;
        %     end
        % elseif all(y_j == [1, 1]') % guided sensing
        e_A = power_cost_HAUV*dur_HAUV_inspect*N_inspect + power_cost_LAUV_scan*dur_LAUV_scan*N_survey; % 1 H-AUV + 1 L-AUV
        e_B = power_cost_LAUV*dur_LAUV_inspect*N_inspect + power_cost_LAUV_scan*dur_LAUV_scan*N_survey; %  L-AUV
        e_C = power_cost_HAUV*(dur_HAUV_inspect*N_inspect + dur_HAUV_scan*N_survey); % H-AUV
        if num_HAUV > 0 && num_LAUV >0 % 1 H-AUV + 1 L-AUV
            task_cost = e_A;
        elseif num_LAUV > 0 %  L-AUV
            task_cost = e_B;
        elseif num_HAUV > 0 %  H-AUV
            task_cost = e_C;
        else 
            task_cost = 0; 
        end
        task_cost_vec(1, j) = task_cost;
        optimal_task_energy_vec(1, j) = e_A;
     end
    % disp([total_travel_cost, sum(task_cost_vec), sum(optimal_task_energy_vec)]);
    tot_energy_consumption = total_travel_cost + sum(task_cost_vec)-sum(optimal_task_energy_vec);

end

