function [reach_NE, total_num_combs] = verify_reach_NE_para(agents, P, Y)
    [~, N_agents] = size(agents);
    [~, N_tasks] = size(P);
    reach_NE = 1; 
    total_num_combs = 0;
    epsilion = 1e-10;

    % syncranization
    for agent_i = 1:N_agents
        for agent_l = 1:N_agents
            agents(agent_i).A(agent_l,:) = agents(agent_l).A(agent_l,:);
        end
    end

    higher_u_vec = zeros(N_agents, 1);
    parfor agent_idx = 1:N_agents
        agent_i = agents(agent_idx); 
        org_utility = utility_calculation(agent_i, P, Y);
        max_num_tasks = agent_i.max_num_tasks;

        C = zeros(1, max_num_tasks);
        for number_selected_tasks = 1:max_num_tasks
            C1 = nchoosek(1:N_tasks, number_selected_tasks);
            C1(:, number_selected_tasks+1:max_num_tasks) = 0;
            C = [C;C1];
        end
        [n_actions, ~] = size(C);
        total_num_combs = total_num_combs + n_actions;
        highest_utility = org_utility;
        best_assignment = agent_i.assignments;
        for a_l = 1:n_actions
            select_task_indices = C(a_l,:);
            select_task_indices = nonzeros(select_task_indices)';
            poss_agent = agent_i;
            assignments = zeros(1,N_tasks);
            assignments(select_task_indices) = 1;
            % poss_agent.theta = assignments;
            % poss_agent.Theta(agent_idx,:) = assignments;
            poss_agent.assignments = assignments;
            poss_agent.A(agent_idx,:) = assignments;
            poss_utility = utility_calculation(poss_agent, P, Y); 
            if any(agent_i.assignments ~= poss_agent.assignments) && (poss_utility-highest_utility) > epsilion
                highest_utility = poss_utility;
                best_assignment = poss_agent.assignments;
                higher_u_vec(agent_idx) = 1;
            end
        end
        if any(agent_i.assignments ~= best_assignment)
            disp([agent_idx, org_utility, highest_utility, (highest_utility-org_utility)/org_utility])
            disp(find(agent_i.assignments))
            disp(find(best_assignment))
        end
    end
    
    if max(higher_u_vec) > 0
        reach_NE = 0;
    else
        reach_NE = 1;
    end 
end 